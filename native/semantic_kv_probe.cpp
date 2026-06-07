#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

struct Args {
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_batch = 1024;
    int32_t n_threads = 10;
    int32_t n_gpu_layers = 0;
};

struct TokenSpan {
    std::vector<llama_token> tokens;
    int32_t p0 = 0;
    int32_t p1 = 0;
};

struct LogitSnapshot {
    std::vector<float> logits;
    std::vector<llama_token> top;
};

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m /path/to/model.gguf [--threads 10] [--ctx 2048]\n", argv0);
}

static Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            args.n_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            args.n_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ngl") == 0 && i + 1 < argc) {
            args.n_gpu_layers = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.model_path.empty()) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_bos) {
    const int n = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n, add_bos, true) < 0) {
        std::fprintf(stderr, "tokenize failed\n");
        std::exit(1);
    }
    return tokens;
}

static void decode_tokens(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    int32_t pos_start,
    llama_seq_id seq_id,
    bool logits_last
) {
    if (tokens.empty()) {
        return;
    }
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
        std::fprintf(stderr, "llama_decode failed with rc=%d\n", rc);
        std::exit(1);
    }
}

static llama_token top_token(llama_context * ctx, const llama_vocab * vocab) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        std::fprintf(stderr, "no logits available\n");
        std::exit(1);
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

static LogitSnapshot snapshot_logits(llama_context * ctx, const llama_vocab * vocab, int32_t top_k) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        std::fprintf(stderr, "no logits available\n");
        std::exit(1);
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

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "";
    }
    return std::string(buf, n);
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

static TokenSpan make_span(
    const llama_vocab * vocab,
    const std::string & text,
    int32_t p0,
    bool add_bos
) {
    TokenSpan span;
    span.tokens = tokenize(vocab, text, add_bos);
    span.p0 = p0;
    span.p1 = p0 + static_cast<int32_t>(span.tokens.size());
    return span;
}

static void print_top_result(const char * name, llama_token token, const llama_vocab * vocab) {
    std::printf(
        "    \"%s\": {\"token_id\": %d, \"piece\": \"%s\"}",
        name,
        token,
        json_escape(token_piece(vocab, token)).c_str());
}

static void print_top_list(const char * name, const LogitSnapshot & snapshot, const llama_vocab * vocab) {
    std::printf("    \"%s\": [", name);
    for (size_t i = 0; i < snapshot.top.size(); ++i) {
        const llama_token token = snapshot.top[i];
        std::printf(
            "%s{\"token_id\": %d, \"piece\": \"%s\", \"logit\": %.6f}",
            i == 0 ? "" : ", ",
            token,
            json_escape(token_piece(vocab, token)).c_str(),
            snapshot.logits[token]);
    }
    std::printf("]");
}

static void print_logit_comparison(const char * name, const LogitSnapshot & left, const LogitSnapshot & right) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    int top_overlap = 0;
    for (size_t i = 0; i < left.logits.size(); ++i) {
        const double l = left.logits[i];
        const double r = right.logits[i];
        const double diff = std::abs(l - r);
        sum_abs += diff;
        sum_sq += diff * diff;
        max_abs = std::max(max_abs, diff);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    for (llama_token token : left.top) {
        if (std::find(right.top.begin(), right.top.end(), token) != right.top.end()) {
            top_overlap += 1;
        }
    }
    const double n = static_cast<double>(left.logits.size());
    const double cosine = dot / std::max(1e-12, std::sqrt(left_norm) * std::sqrt(right_norm));
    std::printf("    \"%s\": {", name);
    std::printf("\"mean_abs_diff\": %.8f, ", sum_abs / n);
    std::printf("\"rms_diff\": %.8f, ", std::sqrt(sum_sq / n));
    std::printf("\"max_abs_diff\": %.8f, ", max_abs);
    std::printf("\"cosine_similarity\": %.8f, ", cosine);
    std::printf("\"top_k_overlap\": %d}", top_overlap);
}

int main(int argc, char ** argv) {
    Args args = parse_args(argc, argv);

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_seq_max = 8;
    ctx_params.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, args.n_threads, args.n_threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);

    const std::string a = "System: answer tersely.\nShared prefix: auth cookie bug report.\n";
    const std::string noise = "Noise: QR scanner camera settings, billing dashboard columns, analytics events.\n";
    const std::string task = "Task: local HTTP rejects SameSite=None without Secure. Expected local cookie is SameSite=Lax. Answer:";

    const TokenSpan a_span = make_span(vocab, a, 0, true);
    const TokenSpan noise_span = make_span(vocab, noise, a_span.p1, false);
    const TokenSpan task_after_a = make_span(vocab, task, a_span.p1, false);
    const TokenSpan task_after_noise = make_span(vocab, task, noise_span.p1, false);

    const llama_seq_id seq_prefix = 0;
    const llama_seq_id seq_branch_noise = 1;
    const llama_seq_id seq_branch_task = 2;
    const llama_seq_id seq_scratch_task = 3;
    const llama_seq_id seq_full = 4;
    const llama_seq_id seq_mutated = 5;

    // Valid branch experiment: compute A once, copy its KV, then extend branches.
    decode_tokens(ctx, a_span.tokens, a_span.p0, seq_prefix, false);
    llama_memory_seq_cp(mem, seq_prefix, seq_branch_noise, -1, -1);
    llama_memory_seq_cp(mem, seq_prefix, seq_branch_task, -1, -1);
    decode_tokens(ctx, noise_span.tokens, noise_span.p0, seq_branch_noise, true);
    decode_tokens(ctx, task_after_a.tokens, task_after_a.p0, seq_branch_task, true);
    const llama_token branch_task_top = top_token(ctx, vocab);
    const LogitSnapshot branch_task_logits = snapshot_logits(ctx, vocab, 5);

    // Scratch A+task baseline.
    decode_tokens(ctx, a_span.tokens, a_span.p0, seq_scratch_task, false);
    decode_tokens(ctx, task_after_a.tokens, task_after_a.p0, seq_scratch_task, true);
    const llama_token scratch_task_top = top_token(ctx, vocab);
    const LogitSnapshot scratch_task_logits = snapshot_logits(ctx, vocab, 5);

    // Invalid middle-removal experiment: compute A+noise+task, copy, remove noise,
    // shift later positions, then recompute only the final task token to get logits.
    decode_tokens(ctx, a_span.tokens, a_span.p0, seq_full, false);
    decode_tokens(ctx, noise_span.tokens, noise_span.p0, seq_full, false);
    decode_tokens(ctx, task_after_noise.tokens, task_after_noise.p0, seq_full, true);
    const llama_token full_top = top_token(ctx, vocab);
    const LogitSnapshot full_logits = snapshot_logits(ctx, vocab, 5);

    llama_memory_seq_cp(mem, seq_full, seq_mutated, -1, -1);
    const bool removed = llama_memory_seq_rm(mem, seq_mutated, noise_span.p0, noise_span.p1);
    llama_memory_seq_add(mem, seq_mutated, noise_span.p1, -1, -static_cast<llama_pos>(noise_span.tokens.size()));

    std::vector<llama_token> last_task_token = { task_after_a.tokens.back() };
    llama_memory_seq_rm(mem, seq_mutated, task_after_a.p1 - 1, -1);
    decode_tokens(ctx, last_task_token, task_after_a.p1 - 1, seq_mutated, true);
    const llama_token mutated_top = top_token(ctx, vocab);
    const LogitSnapshot mutated_logits = snapshot_logits(ctx, vocab, 5);

    std::printf("{\n");
    std::printf("  \"benchmark\": \"semantic_kv_probe\",\n");
    std::printf("  \"tokens\": {\"prefix\": %d, \"noise\": %d, \"task\": %d},\n",
        static_cast<int>(a_span.tokens.size()),
        static_cast<int>(noise_span.tokens.size()),
        static_cast<int>(task_after_a.tokens.size()));
    std::printf("  \"seq_ranges\": {\n");
    std::printf("    \"prefix\": [%d, %d],\n", a_span.p0, a_span.p1);
    std::printf("    \"noise\": [%d, %d],\n", noise_span.p0, noise_span.p1);
    std::printf("    \"task_after_prefix\": [%d, %d]\n", task_after_a.p0, task_after_a.p1);
    std::printf("  },\n");
    std::printf("  \"valid_prefix_branch\": {\n");
    print_top_result("branch_task_top", branch_task_top, vocab);
    std::printf(",\n");
    print_top_result("scratch_task_top", scratch_task_top, vocab);
    std::printf(",\n    \"matches_scratch\": %s\n", branch_task_top == scratch_task_top ? "true" : "false");
    std::printf("    ,\n");
    print_logit_comparison("branch_vs_scratch_logits", branch_task_logits, scratch_task_logits);
    std::printf(",\n");
    print_top_list("branch_top5", branch_task_logits, vocab);
    std::printf(",\n");
    print_top_list("scratch_top5", scratch_task_logits, vocab);
    std::printf("\n");
    std::printf("  },\n");
    std::printf("  \"middle_removal_experiment\": {\n");
    std::printf("    \"seq_rm_returned\": %s,\n", removed ? "true" : "false");
    print_top_result("full_top", full_top, vocab);
    std::printf(",\n");
    print_top_result("mutated_top", mutated_top, vocab);
    std::printf(",\n");
    print_top_result("scratch_task_top", scratch_task_top, vocab);
    std::printf(",\n    \"mutated_matches_scratch\": %s\n", mutated_top == scratch_task_top ? "true" : "false");
    std::printf("    ,\n");
    print_logit_comparison("mutated_vs_scratch_logits", mutated_logits, scratch_task_logits);
    std::printf(",\n");
    print_logit_comparison("full_vs_scratch_logits", full_logits, scratch_task_logits);
    std::printf(",\n");
    print_top_list("mutated_top5", mutated_logits, vocab);
    std::printf(",\n");
    print_top_list("scratch_top5", scratch_task_logits, vocab);
    std::printf("\n");
    std::printf("  },\n");
    std::printf("  \"interpretation\": {\n");
    std::printf("    \"valid_prefix_branch\": \"A shared prefix can be copied to semantic branches with llama_memory_seq_cp.\",\n");
    std::printf("    \"middle_removal_experiment\": \"Removing a middle segment and shifting positions is experimentally possible, but equality with scratch A+task is the correctness test.\"\n");
    std::printf("  }\n");
    std::printf("}\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
