# Architecture

## Objectif

Construire une couche entre l'agent et le modele local :

```text
agent events -> segment store -> scorer -> prompt packer -> LLM runtime
```

La premiere version ne manipule pas directement les tensors KV. Elle cree les conditions qui rendent le KV/prefix cache utile : garder les blocs stables au debut et pousser le volatile en suffixe.

## Concepts

### Event log

Le journal garde tout ce qui s'est passe :

- messages utilisateur ;
- reponses assistant ;
- fichiers lus ;
- commandes ;
- erreurs ;
- decisions.

Rien n'est perdu. "Oublier" signifie seulement retirer un segment du prompt actif.

### Segment store

Chaque bloc utile devient un segment :

```text
id, source, kind, text, token_count, hash, labels, importance, volatility
```

Les labels sont heuristiques en V1. Ils pourront ensuite venir d'embeddings, d'un classifieur local, ou d'un petit modele.

### Scorer

Le scorer favorise :

- les segments proches de la requete courante ;
- les decisions durables ;
- les fichiers stables ;
- les segments recents.

Il penalise :

- le cout token ;
- les logs volatils ;
- les segments hors sujet.

### Prompt packer

Le prompt est reconstruit en trois zones :

```text
stable prefix:
  instructions, objectif, decisions/fichiers stables

dynamic suffix:
  logs, erreurs recentes, dernier message

external memory:
  segments conserves mais non envoyes
```

Sur un moteur comme vLLM, cette forme augmente les chances de reutiliser le prefix cache entre tours.

## Roadmap

These items are planned research directions, not stable API commitments.

1. Mesurer baseline vs routed context sans moteur local.
2. Brancher un client vLLM OpenAI-compatible.
3. Activer Automatic Prefix Caching et logger la latence.
4. Ajouter LMCache pour offload/reuse plus avance.
5. Tester un cache modulaire non-prefix uniquement apres avoir un benchmark fiable.
