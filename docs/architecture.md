# Architecture

## Objectif

Construire une couche entre l'agent et le modele local :

```text
agent events -> segment store -> scorer -> prompt packer -> LLM runtime
```

La premiere version routait seulement le prompt actif. Le projet contient
maintenant aussi un runner natif llama.cpp qui manipule le cache KV au niveau
sequence pour tester des branches partageant un prefixe commun.

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

### Native KV runner

Le runner natif accepte un plan JSON:

```text
segments + ops
```

Operations supportees:

```text
eval
copy
remove
shift
keep
compare
generate
```

Le chemin valide aujourd'hui est:

```text
root context
copy root KV to auth branch
copy root KV to QR branch
continue only the active branch
generate from the copied branch
```

La generation greedy depuis une branche KV copiee a ete comparee a une
evaluation scratch `root + branch + task` et produit le meme texte. Cela valide
le shared-prefix KV branching.

Ce que l'architecture ne valide pas:

```text
A + noise + B -> A + B
```

Le cache KV de `B` n'est pas generalement reutilisable si `B` a ete calcule en
attendant `noise` et avec d'autres positions.

## Roadmap

These items are planned research directions, not stable API commitments.

1. Rejouer une trace agent plus realiste.
2. Mesurer cold-cache, warm prompt-cache, et native branch-cache separement.
3. Ajouter des scenarios de switch de branche avant generation.
4. Mesurer l'empreinte memoire quand le nombre de branches augmente.
5. Tester un cache modulaire non-prefix uniquement apres avoir un benchmark fiable.
