#!/usr/bin/env python3
"""
Generate training data for sign-to-French sentence model.
Uses Claude API to generate (sign_sequence, french_sentence) pairs.

Output: CSV with columns: signs, sentence
  signs: space-separated sign words (e.g., "moi manger pomme")
  sentence: grammatically correct French (e.g., "Je mange une pomme")
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    sys.exit(1)


def load_vocabulary(sl_dir):
    """Load all words from the SL dataset."""
    words = sorted([d for d in os.listdir(sl_dir) if os.path.isdir(os.path.join(sl_dir, d))])
    return words


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


SYSTEM_PROMPT = """Tu es un expert en langue des signes et en français.
Tu vas recevoir une liste de mots de vocabulaire de langue des signes (en français).
Pour chaque batch, tu dois générer des paires d'entraînement pour un modèle de traduction.

Chaque paire contient:
- "signs": les mots tels qu'ils seraient signés (sans articles, sans conjugaison, ordre de la langue des signes)
- "sentence": la phrase française grammaticalement correcte correspondante

RÈGLES IMPORTANTES:
1. Utilise UNIQUEMENT les mots de la liste fournie pour les signes
2. Varie la complexité: phrases simples (2-3 signes) ET complexes (4-7 signes)
3. Varie les structures: affirmatives, négatives, interrogatives, impératives
4. Varie les temps: présent, passé, futur
5. Les signes doivent être en minuscules, séparés par des espaces
6. La phrase française doit être naturelle et correcte
7. Pas de guillemets autour des phrases
8. Génère des phrases du quotidien, réalistes

EXEMPLES:
{"signs": "moi manger pomme", "sentence": "Je mange une pomme"}
{"signs": "hier moi aller ecole", "sentence": "Hier, je suis allé à l'école"}
{"signs": "toi vouloir aide", "sentence": "Est-ce que tu veux de l'aide ?"}
{"signs": "demain nous travailler", "sentence": "Demain, nous travaillerons"}
{"signs": "moi pas comprendre", "sentence": "Je ne comprends pas"}
{"signs": "ou toi habiter", "sentence": "Où est-ce que tu habites ?"}
{"signs": "maman cuisiner bon", "sentence": "Maman cuisine bien"}
{"signs": "hier pluie beaucoup", "sentence": "Hier, il a beaucoup plu"}

Réponds UNIQUEMENT avec un JSON array de objets {"signs": "...", "sentence": "..."}.
Pas de texte avant ou après. Juste le JSON array."""


def generate_batch(client, vocabulary, batch_size=100, used_words=None):
    """Generate a batch of training pairs using Claude API."""
    # Select a diverse subset of words for this batch
    if used_words:
        # Prioritize less-used words
        word_weights = [1.0 / (used_words.get(w, 0) + 1) for w in vocabulary]
        total = sum(word_weights)
        word_weights = [w / total for w in word_weights]
        selected = list(set(random.choices(vocabulary, weights=word_weights, k=min(200, len(vocabulary)))))
    else:
        selected = random.sample(vocabulary, min(200, len(vocabulary)))

    prompt = f"""Voici {len(selected)} mots de vocabulaire disponibles:
{', '.join(selected)}

Génère exactement {batch_size} paires d'entraînement variées utilisant ces mots.
Assure-toi de varier:
- La longueur (2 à 7 signes par phrase)
- Les structures grammaticales
- Les temps verbaux
- Les types de phrases (affirmative, négative, interrogative)

JSON array uniquement:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Parse JSON
        if text.startswith('['):
            pairs = json.loads(text)
        else:
            # Try to extract JSON from response
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                pairs = json.loads(text[start:end])
            else:
                print(f"  WARNING: Could not parse response")
                return []

        # Validate pairs
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and 'signs' in p and 'sentence' in p:
                signs = p['signs'].strip().lower()
                sentence = p['sentence'].strip()
                if signs and sentence and len(signs.split()) >= 2:
                    valid_pairs.append({'signs': signs, 'sentence': sentence})

        return valid_pairs

    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl-dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'SL'),
                        help='Path to SL dataset directory')
    parser.add_argument('--output', default='training_pairs.csv', help='Output CSV file')
    parser.add_argument('--target', type=int, default=70000, help='Target number of pairs')
    parser.add_argument('--batch-size', type=int, default=80, help='Pairs per API call')
    parser.add_argument('--resume', action='store_true', help='Resume from existing file')
    args = parser.parse_args()

    # Find SL directory
    sl_dir = os.path.abspath(args.sl_dir)
    if not os.path.isdir(sl_dir):
        # Try common locations
        for candidate in [
            '/Users/alopro/Desktop/AI/RECHERCHE/SignInterpreter/SL',
            os.path.join(os.path.dirname(__file__), '..', '..', 'SL'),
        ]:
            if os.path.isdir(candidate):
                sl_dir = candidate
                break
        else:
            print(f"ERROR: SL directory not found. Use --sl-dir")
            sys.exit(1)

    vocabulary = load_vocabulary(sl_dir)
    print(f"Vocabulary: {len(vocabulary)} words from {sl_dir}")

    # Initialize Claude client
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load existing pairs if resuming
    existing_pairs = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_pairs = list(reader)
        print(f"Resuming: {len(existing_pairs)} existing pairs")

    # Track word usage for diversity
    used_words = {}
    for p in existing_pairs:
        for w in p['signs'].split():
            used_words[w] = used_words.get(w, 0) + 1

    total_pairs = len(existing_pairs)
    all_pairs = list(existing_pairs)

    # Calculate batches needed
    remaining = args.target - total_pairs
    n_batches = (remaining + args.batch_size - 1) // args.batch_size

    print(f"Target: {args.target} pairs")
    print(f"Remaining: {remaining}")
    print(f"Batches: {n_batches} x {args.batch_size}")
    print(f"Output: {args.output}")
    print()

    for batch_num in range(n_batches):
        if total_pairs >= args.target:
            break

        print(f"  Batch {batch_num + 1}/{n_batches} ({total_pairs}/{args.target})...", end=' ', flush=True)

        pairs = generate_batch(client, vocabulary, args.batch_size, used_words)

        if pairs:
            all_pairs.extend(pairs)
            total_pairs += len(pairs)

            # Update word usage
            for p in pairs:
                for w in p['signs'].split():
                    used_words[w] = used_words.get(w, 0) + 1

            print(f"+{len(pairs)} = {total_pairs}")

            # Save after each batch (resume-safe)
            with open(args.output, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['signs', 'sentence'])
                writer.writeheader()
                writer.writerows(all_pairs)
        else:
            print("FAILED")

        # Rate limiting
        time.sleep(1)

    print(f"\nDone! Total: {total_pairs} pairs saved to {args.output}")

    # Stats
    avg_signs = sum(len(p['signs'].split()) for p in all_pairs) / len(all_pairs) if all_pairs else 0
    avg_sentence = sum(len(p['sentence'].split()) for p in all_pairs) / len(all_pairs) if all_pairs else 0
    unique_words_used = len(used_words)
    print(f"Avg signs/pair: {avg_signs:.1f}")
    print(f"Avg words/sentence: {avg_sentence:.1f}")
    print(f"Unique vocabulary used: {unique_words_used}/{len(vocabulary)}")


if __name__ == "__main__":
    main()
