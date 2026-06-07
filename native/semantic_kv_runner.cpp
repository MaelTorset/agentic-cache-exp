#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const Json & at(const std::string & key) const {
        auto it = object.find(key);
        if (it == object.end()) {
            throw std::runtime_error("missing key: " + key);
        }
        return it->second;
    }

    const Json * find(const std::string & key) const {
        auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string input) : input(std::move(input)) {}

    Json parse() {
        Json value = parse_value();
        skip_ws();
        if (pos != input.size()) {
            fail("unexpected trailing data");
        }
        return value;
    }

private:
    std::string input;
    size_t pos = 0;

    [[noreturn]] void fail(const std::string & message) const {
        throw std::runtime_error("json parse error at byte " + std::to_string(pos) + ": " + message);
    }

    void skip_ws() {
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\n' || input[pos] == '\r' || input[pos] == '\t')) {
            pos++;
        }
    }

    char peek() {
        skip_ws();
        if (pos >= input.size()) {
            fail("unexpected end of input");
        }
        return input[pos];
    }

    bool consume(char ch) {
        skip_ws();
        if (pos < input.size() && input[pos] == ch) {
            pos++;
            return true;
        }
        return false;
    }

    Json parse_value() {
        const char ch = peek();
        if (ch == '{') return parse_object();
        if (ch == '[') return parse_array();
        if (ch == '"') return parse_string();
        if (ch == '-' || (ch >= '0' && ch <= '9')) return parse_number();
        if (input.compare(pos, 4, "true") == 0) {
            pos += 4;
            Json value;
            value.type = Json::Type::Bool;
            value.boolean = true;
            return value;
        }
        if (input.compare(pos, 5, "false") == 0) {
            pos += 5;
            Json value;
            value.type = Json::Type::Bool;
            value.boolean = false;
            return value;
        }
        if (input.compare(pos, 4, "null") == 0) {
            pos += 4;
            return Json{};
        }
        fail("invalid value");
    }

    Json parse_object() {
        if (!consume('{')) fail("expected object");
        Json value;
        value.type = Json::Type::Object;
        if (consume('}')) return value;
        while (true) {
            Json key = parse_string();
            if (!consume(':')) fail("expected ':' after object key");
            value.object.emplace(key.string, parse_value());
            if (consume('}')) return value;
            if (!consume(',')) fail("expected ',' or '}' in object");
        }
    }

    Json parse_array() {
        if (!consume('[')) fail("expected array");
        Json value;
        value.type = Json::Type::Array;
        if (consume(']')) return value;
        while (true) {
            value.array.push_back(parse_value());
            if (consume(']')) return value;
            if (!consume(',')) fail("expected ',' or ']' in array");
        }
    }

    Json parse_string() {
        if (!consume('"')) fail("expected string");
        Json value;
        value.type = Json::Type::String;
        while (pos < input.size()) {
            char ch = input[pos++];
            if (ch == '"') return value;
            if (ch != '\\') {
                value.string += ch;
                continue;
            }
            if (pos >= input.size()) fail("unterminated escape");
            char esc = input[pos++];
            switch (esc) {
                case '"': value.string += '"'; break;
                case '\\': value.string += '\\'; break;
                case '/': value.string += '/'; break;
                case 'b': value.string += '\b'; break;
                case 'f': value.string += '\f'; break;
                case 'n': value.string += '\n'; break;
                case 'r': value.string += '\r'; break;
                case 't': value.string += '\t'; break;
                default: fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }

    Json parse_number() {
        skip_ws();
        const size_t start = pos;
        if (input[pos] == '-') pos++;
        while (pos < input.size() && input[pos] >= '0' && input[pos] <= '9') pos++;
        if (pos < input.size() && input[pos] == '.') {
            pos++;
            while (pos < input.size() && input[pos] >= '0' && input[pos] <= '9') pos++;
        }
        if (pos < input.size() && (input[pos] == 'e' || input[pos] == 'E')) {
            pos++;
            if (pos < input.size() && (input[pos] == '+' || input[pos] == '-')) pos++;
            while (pos < input.size() && input[pos] >= '0' && input[pos] <= '9') pos++;
        }
        Json value;
        value.type = Json::Type::Number;
        value.number = std::stod(input.substr(start, pos - start));
        return value;
    }
};

struct Args {
    std::string model_path;
    std::string plan_path;
    int32_t n_ctx = 2048;
    int32_t n_batch = 1024;
    int32_t n_threads = 10;
    int32_t n_gpu_layers = 0;
};

struct Range {
    int32_t p0 = 0;
    int32_t p1 = 0;
};

struct Segment {
    std::string id;
    std::string text;
    bool add_bos = false;
    std::vector<llama_token> tokens;
};

struct LogitSnapshot {
    std::vector<float> logits;
    std::vector<llama_token> top;
};

struct SequenceState {
    int32_t next_pos = 0;
    int32_t tokens_evaled = 0;
    std::map<std::string, Range> ranges;
    bool has_logits = false;
    LogitSnapshot logits;
};

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m model.gguf --plan plan.json [--threads 10] [--ctx 2048] [--batch 1024]\n", argv0);
}

static Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--plan") == 0 && i + 1 < argc) {
            args.plan_path = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            args.n_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            args.n_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ngl") == 0 && i + 1 < argc) {
            args.n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.model_path.empty() || args.plan_path.empty()) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

static std::string read_file(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

static std::string json_escape(const std::string & raw) {
    std::string out;
    for (char ch : raw) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

static std::string get_string(const Json & object, const std::string & key) {
    const Json & value = object.at(key);
    if (!value.is_string()) {
        throw std::runtime_error("expected string key: " + key);
    }
    return value.string;
}

static std::string get_string_default(const Json & object, const std::string & key, const std::string & fallback) {
    const Json * value = object.find(key);
    if (value == nullptr) return fallback;
    if (!value->is_string()) throw std::runtime_error("expected string key: " + key);
    return value->string;
}

static int32_t get_int(const Json & object, const std::string & key) {
    const Json & value = object.at(key);
    if (!value.is_number()) {
        throw std::runtime_error("expected number key: " + key);
    }
    return static_cast<int32_t>(value.number);
}

static int32_t get_int_default(const Json & object, const std::string & key, int32_t fallback) {
    const Json * value = object.find(key);
    if (value == nullptr) return fallback;
    if (!value->is_number()) throw std::runtime_error("expected number key: " + key);
    return static_cast<int32_t>(value->number);
}

static bool get_bool_default(const Json & object, const std::string & key, bool fallback) {
    const Json * value = object.find(key);
    if (value == nullptr) return fallback;
    if (!value->is_bool()) throw std::runtime_error("expected bool key: " + key);
    return value->boolean;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_bos) {
    const int n = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n, add_bos, true) < 0) {
        throw std::runtime_error("tokenization failed");
    }
    return tokens;
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

static void decode_tokens(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    int32_t pos_start,
    llama_seq_id seq_id,
    bool logits_last
) {
    if (tokens.empty()) return;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i] = logits_last && i == batch.n_tokens - 1 ? 1 : 0;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with rc=" + std::to_string(rc));
    }
}

static LogitSnapshot snapshot_logits(llama_context * ctx, const llama_vocab * vocab, int32_t top_k) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("no logits available");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    LogitSnapshot snapshot;
    snapshot.logits.assign(logits, logits + n_vocab);
    snapshot.top.resize(n_vocab);
    std::iota(snapshot.top.begin(), snapshot.top.end(), 0);
    const int32_t keep = std::min<int32_t>(top_k, n_vocab);
    std::partial_sort(
        snapshot.top.begin(),
        snapshot.top.begin() + keep,
        snapshot.top.end(),
        [&](llama_token left, llama_token right) {
            return snapshot.logits[left] > snapshot.logits[right];
        });
    snapshot.top.resize(keep);
    return snapshot;
}

static double elapsed_ms(std::chrono::steady_clock::time_point started) {
    const auto ended = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(ended - started).count();
}

static Range require_range(const SequenceState & state, const std::string & segment) {
    auto it = state.ranges.find(segment);
    if (it == state.ranges.end()) {
        throw std::runtime_error("sequence is missing segment range: " + segment);
    }
    return it->second;
}

static int32_t resolve_position(const Json & op, const SequenceState & state, const std::string & direct_key) {
    const Json * direct = op.find(direct_key);
    if (direct != nullptr) {
        if (!direct->is_number()) throw std::runtime_error("expected numeric position: " + direct_key);
        return static_cast<int32_t>(direct->number);
    }
    const Json * start_after = op.find("start_after_segment");
    if (start_after != nullptr) {
        if (!start_after->is_string()) throw std::runtime_error("expected string start_after_segment");
        return require_range(state, start_after->string).p1;
    }
    return -1;
}

static int32_t resolve_delta(const Json & op, const SequenceState & state) {
    const Json * direct = op.find("delta");
    if (direct != nullptr) {
        if (!direct->is_number()) throw std::runtime_error("expected numeric delta");
        return static_cast<int32_t>(direct->number);
    }
    const Json * delta_segment = op.find("delta_segment");
    if (delta_segment == nullptr || !delta_segment->is_string()) {
        throw std::runtime_error("shift op requires delta or delta_segment");
    }
    const Range range = require_range(state, delta_segment->string);
    const std::string direction = get_string_default(op, "direction", "negative");
    const int32_t length = range.p1 - range.p0;
    return direction == "positive" ? length : -length;
}

static void apply_shift_to_state(SequenceState & state, int32_t p0, int32_t p1, int32_t delta) {
    if (p1 < 0) {
        state.next_pos += delta;
    }
    for (auto & [_, range] : state.ranges) {
        if (range.p0 >= p0 && (p1 < 0 || range.p0 < p1)) {
            range.p0 += delta;
            range.p1 += delta;
        }
    }
}

struct Comparison {
    std::string label;
    bool top_token_match = false;
    double mean_abs_diff = 0.0;
    double rms_diff = 0.0;
    double max_abs_diff = 0.0;
    double cosine_similarity = 0.0;
    int32_t top_k_overlap = 0;
    llama_token left_top = 0;
    llama_token right_top = 0;
};

static Comparison compare_logits(const std::string & label, const LogitSnapshot & left, const LogitSnapshot & right) {
    Comparison comparison;
    comparison.label = label;
    comparison.left_top = left.top.empty() ? 0 : left.top[0];
    comparison.right_top = right.top.empty() ? 0 : right.top[0];
    comparison.top_token_match = comparison.left_top == comparison.right_top;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < left.logits.size(); ++i) {
        const double l = left.logits[i];
        const double r = right.logits[i];
        const double diff = std::abs(l - r);
        sum_abs += diff;
        sum_sq += diff * diff;
        comparison.max_abs_diff = std::max(comparison.max_abs_diff, diff);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    for (llama_token token : left.top) {
        if (std::find(right.top.begin(), right.top.end(), token) != right.top.end()) {
            comparison.top_k_overlap += 1;
        }
    }
    const double n = static_cast<double>(left.logits.size());
    comparison.mean_abs_diff = sum_abs / n;
    comparison.rms_diff = std::sqrt(sum_sq / n);
    comparison.cosine_similarity = dot / std::max(1e-12, std::sqrt(left_norm) * std::sqrt(right_norm));
    return comparison;
}

static void print_error(const std::string & code, const std::string & message) {
    std::printf("{\"error\":\"%s\",\"message\":\"%s\"}\n", json_escape(code).c_str(), json_escape(message).c_str());
}

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        const Json plan = JsonParser(read_file(args.plan_path)).parse();
        if (!plan.is_object()) throw std::runtime_error("plan root must be an object");

        const Json * config = plan.find("config");
        const int32_t top_k = config && config->is_object() ? get_int_default(*config, "top_k", 5) : 5;
        const bool suppress_logs = config && config->is_object() ? get_bool_default(*config, "suppress_logs", true) : true;
        if (suppress_logs) {
            llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
        }
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = args.n_gpu_layers;
        llama_model * model = llama_model_load_from_file(args.model_path.c_str(), model_params);
        if (model == nullptr) throw std::runtime_error("failed to load model");

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = args.n_ctx;
        ctx_params.n_batch = args.n_batch;
        ctx_params.n_seq_max = 16;
        ctx_params.no_perf = false;
        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) throw std::runtime_error("failed to create context");
        llama_set_n_threads(ctx, args.n_threads, args.n_threads);

        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_memory_t mem = llama_get_memory(ctx);

        std::map<std::string, Segment> segments;
        const Json & segments_json = plan.at("segments");
        if (!segments_json.is_array()) throw std::runtime_error("segments must be an array");
        for (const Json & item : segments_json.array) {
            if (!item.is_object()) throw std::runtime_error("segment must be an object");
            Segment segment;
            segment.id = get_string(item, "id");
            segment.text = get_string(item, "text");
            segment.add_bos = get_bool_default(item, "add_bos", false);
            segment.tokens = tokenize(vocab, segment.text, segment.add_bos);
            segments.emplace(segment.id, std::move(segment));
        }

        std::map<int32_t, SequenceState> sequences;
        std::vector<std::string> op_outputs;
        std::vector<Comparison> comparisons;
        const Json & ops = plan.at("ops");
        if (!ops.is_array()) throw std::runtime_error("ops must be an array");

        for (size_t index = 0; index < ops.array.size(); ++index) {
            const Json & op = ops.array[index];
            if (!op.is_object()) throw std::runtime_error("op must be an object");
            const std::string type = get_string(op, "op");
            auto started = std::chrono::steady_clock::now();
            std::ostringstream item;
            item << "{\"index\":" << index << ",\"op\":\"" << json_escape(type) << "\"";

            if (type == "eval") {
                const int32_t seq = get_int(op, "seq");
                const std::string segment_id = get_string(op, "segment");
                auto segment_it = segments.find(segment_id);
                if (segment_it == segments.end()) throw std::runtime_error("unknown segment: " + segment_id);
                SequenceState & state = sequences[seq];
                const int32_t resolved_pos = resolve_position(op, state, "pos");
                const int32_t pos = resolved_pos >= 0 ? resolved_pos : state.next_pos;
                const bool logits = get_bool_default(op, "logits", false);
                decode_tokens(ctx, segment_it->second.tokens, pos, seq, logits);
                const int32_t p1 = pos + static_cast<int32_t>(segment_it->second.tokens.size());
                state.ranges[segment_id] = Range{pos, p1};
                state.next_pos = std::max(state.next_pos, p1);
                state.tokens_evaled += static_cast<int32_t>(segment_it->second.tokens.size());
                if (logits) {
                    state.logits = snapshot_logits(ctx, vocab, top_k);
                    state.has_logits = true;
                }
                item << ",\"seq\":" << seq << ",\"segment\":\"" << json_escape(segment_id) << "\"";
                item << ",\"p0\":" << pos << ",\"p1\":" << p1 << ",\"tokens\":" << segment_it->second.tokens.size();
            } else if (type == "copy") {
                const int32_t from = get_int(op, "from");
                const int32_t to = get_int(op, "to");
                llama_memory_seq_cp(mem, from, to, -1, -1);
                sequences[to] = sequences[from];
                item << ",\"from\":" << from << ",\"to\":" << to;
            } else if (type == "remove") {
                const int32_t seq = get_int(op, "seq");
                SequenceState & state = sequences[seq];
                int32_t p0 = -1;
                int32_t p1 = -1;
                const Json * segment = op.find("segment");
                if (segment != nullptr) {
                    if (!segment->is_string()) throw std::runtime_error("remove segment must be string");
                    const Range range = require_range(state, segment->string);
                    p0 = range.p0;
                    p1 = range.p1;
                } else {
                    p0 = get_int_default(op, "p0", -1);
                    p1 = get_int_default(op, "p1", -1);
                }
                const bool ok = llama_memory_seq_rm(mem, seq, p0, p1);
                item << ",\"seq\":" << seq << ",\"p0\":" << p0 << ",\"p1\":" << p1 << ",\"seq_rm_returned\":" << (ok ? "true" : "false");
            } else if (type == "shift") {
                const int32_t seq = get_int(op, "seq");
                SequenceState & state = sequences[seq];
                const int32_t p0 = resolve_position(op, state, "p0");
                const int32_t p1 = get_int_default(op, "p1", -1);
                const int32_t delta = resolve_delta(op, state);
                llama_memory_seq_add(mem, seq, p0, p1, delta);
                apply_shift_to_state(state, p0, p1, delta);
                item << ",\"seq\":" << seq << ",\"p0\":" << p0 << ",\"p1\":" << p1 << ",\"delta\":" << delta;
            } else if (type == "keep") {
                const int32_t seq = get_int(op, "seq");
                llama_memory_seq_keep(mem, seq);
                for (auto it = sequences.begin(); it != sequences.end();) {
                    if (it->first == seq) ++it;
                    else it = sequences.erase(it);
                }
                item << ",\"seq\":" << seq;
            } else if (type == "compare") {
                const int32_t left = get_int(op, "left");
                const int32_t right = get_int(op, "right");
                const std::string label = get_string(op, "label");
                if (!sequences[left].has_logits || !sequences[right].has_logits) {
                    throw std::runtime_error("compare requires captured logits on both sequences");
                }
                comparisons.push_back(compare_logits(label, sequences[left].logits, sequences[right].logits));
                item << ",\"left\":" << left << ",\"right\":" << right << ",\"label\":\"" << json_escape(label) << "\"";
            } else {
                throw std::runtime_error("unknown op: " + type);
            }

            item << ",\"latency_ms\":" << elapsed_ms(started) << ",\"ok\":true}";
            op_outputs.push_back(item.str());
        }

        std::printf("{\n");
        std::printf("  \"benchmark\":\"semantic_kv_plan\",\n");
        std::printf("  \"model\":\"%s\",\n", json_escape(args.model_path).c_str());
        std::printf("  \"config\":{\"top_k\":%d,\"ctx\":%d,\"threads\":%d},\n", top_k, args.n_ctx, args.n_threads);
        std::printf("  \"segments\":{");
        bool first = true;
        for (const auto & [id, segment] : segments) {
            std::printf("%s\"%s\":{\"tokens\":%zu}", first ? "" : ",", json_escape(id).c_str(), segment.tokens.size());
            first = false;
        }
        std::printf("},\n");
        std::printf("  \"ops\":[");
        for (size_t i = 0; i < op_outputs.size(); ++i) {
            std::printf("%s%s", i == 0 ? "" : ",", op_outputs[i].c_str());
        }
        std::printf("],\n");
        std::printf("  \"sequences\":{");
        first = true;
        for (const auto & [seq, state] : sequences) {
            const llama_pos pos_min = llama_memory_seq_pos_min(mem, seq);
            const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq);
            std::printf(
                "%s\"%d\":{\"pos_min\":%d,\"pos_max\":%d,\"next_pos\":%d,\"tokens_evaled\":%d}",
                first ? "" : ",",
                seq,
                static_cast<int>(pos_min),
                static_cast<int>(pos_max),
                state.next_pos,
                state.tokens_evaled);
            first = false;
        }
        std::printf("},\n");
        std::printf("  \"comparisons\":{");
        for (size_t i = 0; i < comparisons.size(); ++i) {
            const Comparison & c = comparisons[i];
            std::printf(
                "%s\"%s\":{\"top_token_match\":%s,\"mean_abs_diff\":%.8f,\"rms_diff\":%.8f,\"max_abs_diff\":%.8f,\"cosine_similarity\":%.8f,\"top_k_overlap\":%d,\"left_top\":%d,\"right_top\":%d}",
                i == 0 ? "" : ",",
                json_escape(c.label).c_str(),
                c.top_token_match ? "true" : "false",
                c.mean_abs_diff,
                c.rms_diff,
                c.max_abs_diff,
                c.cosine_similarity,
                c.top_k_overlap,
                c.left_top,
                c.right_top);
        }
        std::printf("}\n");
        std::printf("}\n");

        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & exc) {
        print_error("semantic_kv_runner_failed", exc.what());
        return 1;
    }
}
