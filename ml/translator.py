"""
Sentence translator: sign words → grammatically correct French.

Primary: neuro-symbolic BiLSTM slot-filler (data/slot_model/)
Fallback: deterministic rule-based grammar (always available)
"""

import os
import re

# ── Rule-based translation tables ─────────────────────────────────────────────

_SUBJECTS = {'moi', 'tu', 'nous', 'ami', 'bebe', 'chat'}
_VERBS    = {'manger', 'boire', 'aller', 'apprendre', 'comprendre',
             'changer', 'aimer', 'vouloir', 'donner', 'attendre'}
_OBJECTS  = {'aide', 'ami', 'amour', 'bebe', 'chat', 'moi', 'tu', 'nous'}
_SOCIAL   = {'bonjour', 'au revoir', 'merci', 'oui', 'non', 'comment', 'bien', 'bon'}
_MODS     = {'bien', 'bon', 'ici', 'bientot', 'aujourd hui'}

# Pronoun forms: (before_consonant, before_vowel)
_PRON = {
    'moi':  ('je',      "j'"),
    'tu':   ('tu',      'tu'),
    'nous': ('nous',    'nous'),
    'ami':  ('mon ami', 'mon ami'),
    'bebe': ('le bébé', 'le bébé'),
    'chat': ('le chat', 'le chat'),
}

# Conjugations keyed by subject sign word
_CONJ = {
    'manger':    {'moi': 'mange',    'tu': 'manges',   'nous': 'mangeons',  'ami': 'mange',   'bebe': 'mange',   'chat': 'mange'},
    'boire':     {'moi': 'bois',     'tu': 'bois',     'nous': 'buvons',    'ami': 'boit',    'bebe': 'boit',    'chat': 'boit'},
    'aller':     {'moi': 'pars',     'tu': 'pars',     'nous': 'partons',   'ami': 'part',    'bebe': 'part',    'chat': 'part'},
    'apprendre': {'moi': 'apprends', 'tu': 'apprends', 'nous': 'apprenons', 'ami': 'apprend', 'bebe': 'apprend', 'chat': 'apprend'},
    'comprendre':{'moi': 'comprends','tu': 'comprends', 'nous': 'comprenons','ami': 'comprend','bebe': 'comprend','chat': 'comprend'},
    'changer':   {'moi': 'change',   'tu': 'changes',  'nous': 'changeons', 'ami': 'change',  'bebe': 'change',  'chat': 'change'},
    'aimer':     {'moi': 'aime',     'tu': 'aimes',    'nous': 'aimons',    'ami': 'aime',    'bebe': 'aime',    'chat': 'aime'},
    'vouloir':   {'moi': 'veux',     'tu': 'veux',     'nous': 'voulons',   'ami': 'veut',    'bebe': 'veut',    'chat': 'veut'},
    'donner':    {'moi': 'donne',    'tu': 'donnes',   'nous': 'donnons',   'ami': 'donne',   'bebe': 'donne',   'chat': 'donne'},
    'attendre':  {'moi': 'attends',  'tu': 'attends',  'nous': 'attendons', 'ami': 'attend',  'bebe': 'attend',  'chat': 'attend'},
}

_NEG_CONJ = {
    'manger':    {'moi': 'mange pas',    'tu': 'manges pas',   'nous': 'mangeons pas',  'ami': 'mange pas',   'bebe': 'mange pas',   'chat': 'mange pas'},
    'boire':     {'moi': 'bois pas',     'tu': 'bois pas',     'nous': 'buvons pas',    'ami': 'boit pas',    'bebe': 'boit pas',    'chat': 'boit pas'},
    'aller':     {'moi': 'pars pas',     'tu': 'pars pas',     'nous': 'partons pas',   'ami': 'part pas',    'bebe': 'part pas',    'chat': 'part pas'},
    'apprendre': {'moi': "apprends pas", 'tu': "apprends pas", 'nous': "apprenons pas", 'ami': "apprend pas", 'bebe': "apprend pas", 'chat': "apprend pas"},
    'comprendre':{'moi': "comprends pas",'tu': "comprends pas", 'nous': "comprenons pas",'ami': "comprend pas",'bebe': "comprend pas",'chat': "comprend pas"},
    'changer':   {'moi': 'change pas',   'tu': 'changes pas',  'nous': 'changeons pas', 'ami': 'change pas',  'bebe': 'change pas',  'chat': 'change pas'},
    'aimer':     {'moi': 'aime pas',     'tu': 'aimes pas',    'nous': 'aimons pas',    'ami': 'aime pas',    'bebe': 'aime pas',    'chat': 'aime pas'},
    'vouloir':   {'moi': 'veux pas',     'tu': 'veux pas',     'nous': 'voulons pas',   'ami': 'veut pas',    'bebe': 'veut pas',    'chat': 'veut pas'},
    'donner':    {'moi': 'donne pas',    'tu': 'donnes pas',   'nous': 'donnons pas',   'ami': 'donne pas',   'bebe': 'donne pas',   'chat': 'donne pas'},
    'attendre':  {'moi': 'attends pas',  'tu': 'attends pas',  'nous': 'attendons pas', 'ami': 'attend pas',  'bebe': 'attend pas',  'chat': 'attend pas'},
}

_OBJ_FR = {
    'aide':  "de l'aide",
    'ami':   'mon ami',
    'amour': "de l'amour",
    'bebe':  'le bébé',
    'chat':  'le chat',
    'moi':   'moi',
    'tu':    'toi',
    'nous':  'nous',
}

_AVEC_FR = {
    'ami':  'mon ami',
    'moi':  'moi',
    'tu':   'toi',
    'nous': 'nous',
    'bebe': 'le bébé',
    'chat': 'le chat',
}

_VOWELS = set('aeiouâêîôûàéèùœ')


def _vowel_start(s: str) -> bool:
    return bool(s) and s[0].lower() in _VOWELS


def _subject_fr(subj: str, next_word: str = '') -> str:
    """Return French subject pronoun with elision if needed."""
    con, vow = _PRON.get(subj, (subj, subj))
    if subj in ('moi', 'tu', 'nous'):
        return vow if _vowel_start(next_word) else con
    return con


def _apply_neg(subj: str, verb_fr: str, obj_fr: str = '') -> str:
    """Wrap verb phrase with ne/n' ... pas elision."""
    subj_fr = _subject_fr(subj, 'ne')
    particle = "n'" if _vowel_start(verb_fr) else 'ne '
    neg_verb = f"{particle}{verb_fr}"
    if obj_fr:
        # object after negative: de l'aide → pas d'aide
        obj_neg = obj_fr.replace("de l’", "pas d’").replace("de l'", "pas d'")
        if obj_neg == obj_fr:
            obj_neg = 'pas ' + obj_fr
        return f"{subj_fr} {neg_verb} {obj_neg}"
    return f"{subj_fr} {neg_verb}"


def rule_translate(signs_list: list) -> str:
    """Deterministic rule-based LSF→French for the 30-word vocabulary."""
    signs = [s.lower() for s in signs_list if s]
    if not signs:
        return ''

    # Tokenise multi-word signs
    joined = ' '.join(signs)
    # Replace multi-word tokens so they stay together
    for mw in ('au revoir', 'aujourd hui'):
        joined = joined.replace(mw.replace(' ', ' '), mw)
    tokens = joined.split()

    # Re-merge multi-word tokens that were split
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == 'au' and tokens[i + 1] == 'revoir':
            merged.append('au revoir')
            i += 2
        elif i + 1 < len(tokens) and tokens[i] == 'aujourd' and tokens[i + 1] == 'hui':
            merged.append("aujourd hui")
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    toks = merged

    # ── Social / standalone openers ───────────────────────────────────────
    greeting_fr = []
    body_toks = []
    in_greeting = True
    for t in toks:
        if in_greeting and t in ('bonjour', 'au revoir', 'merci'):
            mapping = {'bonjour': 'Bonjour', 'au revoir': 'Au revoir', 'merci': 'Merci'}
            greeting_fr.append(mapping[t])
        elif in_greeting and t == 'oui':
            greeting_fr.append('Oui')
        elif in_greeting and t == 'non' and not body_toks:
            greeting_fr.append('Non')
        else:
            in_greeting = False
            body_toks.append(t)

    prefix = (' '.join(greeting_fr) + ' ') if greeting_fr else ''

    if not body_toks:
        # Only social words → exclamation
        phrase = ' '.join(greeting_fr) + ' !'
        return phrase[0].upper() + phrase[1:] if phrase else ''

    # ── Find subject ──────────────────────────────────────────────────────
    subj = None
    subj_idx = -1
    for idx, t in enumerate(body_toks):
        if t in _SUBJECTS:
            subj = t
            subj_idx = idx
            break

    # ── Temporal prefix (aujourd hui anywhere) ────────────────────────────
    temporal_prefix = ''
    working = list(body_toks)
    if 'aujourd hui' in working:
        temporal_prefix = "Aujourd'hui "
        working = [t for t in working if t != 'aujourd hui']
        # re-find subject
        subj = None
        subj_idx = -1
        for idx, t in enumerate(working):
            if t in _SUBJECTS:
                subj = t
                subj_idx = idx
                break

    # ── Comment (question) ────────────────────────────────────────────────
    is_question = working and working[0] == 'comment'
    if is_question:
        working = working[1:]
        if not working:
            return 'Comment ?'
        subj = None
        subj_idx = -1
        for idx, t in enumerate(working):
            if t in _SUBJECTS:
                subj = t
                subj_idx = idx
                break

    # ── Negation ──────────────────────────────────────────────────────────
    negated = False
    if 'non' in working:
        neg_idx = working.index('non')
        # Only treat as negation if it's before the verb
        verb_in_working = next((t for t in working if t in _VERBS), None)
        if verb_in_working:
            verb_idx_w = working.index(verb_in_working)
            if neg_idx < verb_idx_w or neg_idx == verb_idx_w - 1:
                negated = True
                working = [t for i, t in enumerate(working) if i != neg_idx]

    # ── Find verb ─────────────────────────────────────────────────────────
    verb = None
    verb_idx = -1
    for idx, t in enumerate(working):
        if t in _VERBS:
            verb = t
            verb_idx = idx
            break

    # ── avec + person ─────────────────────────────────────────────────────
    avec_person = None
    if 'avec' in working:
        av_idx = working.index('avec')
        if av_idx + 1 < len(working) and working[av_idx + 1] in _AVEC_FR:
            avec_person = working[av_idx + 1]
            working = [t for i, t in enumerate(working) if i != av_idx and i != av_idx + 1]
            # re-find verb after removal
            verb = None
            verb_idx = -1
            for idx, t in enumerate(working):
                if t in _VERBS:
                    verb = t
                    verb_idx = idx
                    break

    # ── Infinitive verb object (vouloir/aimer + verb) ────────────────────
    inf_verb = None
    _INF_FR = {
        'manger': 'manger', 'boire': 'boire', 'aller': 'aller',
        'apprendre': 'apprendre', 'comprendre': 'comprendre', 'changer': 'changer',
        'aimer': 'aimer', 'vouloir': 'vouloir', 'donner': 'donner', 'attendre': 'attendre',
    }
    if verb_idx >= 0:
        after_verb = working[verb_idx + 1:]
        for t in after_verb:
            if t in _VERBS:
                inf_verb = _INF_FR[t]
                break

    # ── Objects after verb (non-verb tokens) ─────────────────────────────
    objects = []
    if verb_idx >= 0:
        after_verb = working[verb_idx + 1:]
        for t in after_verb:
            if t in _OBJ_FR and t != subj and t not in _VERBS:
                objects.append(t)

    # ── Modifiers ─────────────────────────────────────────────────────────
    mods = []
    for t in working:
        if t in ('bien', 'bon'):
            mods.append('bien')
        elif t == 'ici':
            mods.append('ici')
        elif t == 'bientot':
            mods.append('bientôt')

    # ── No verb found — simple noun/adjective phrase ──────────────────────
    if verb is None:
        if not subj and not working:
            return (prefix or temporal_prefix).rstrip(', ').rstrip() + '.'
        parts = []
        for t in working:
            if t in _OBJ_FR:
                parts.append(_OBJ_FR[t])
            elif t in ('bien', 'bon'):
                parts.append('bien')
            elif t == 'ici':
                parts.append('ici')
            elif t == 'bientot':
                parts.append('bientôt')
            elif t == "aujourd hui":
                parts.append("aujourd'hui")
            elif t in _SUBJECTS:
                pron, _ = _PRON.get(t, (t, t))
                parts.append(pron)
            elif t not in ('non', 'avec', 'oui'):
                parts.append(t)
        sentence = (prefix or temporal_prefix) + ' '.join(parts)
        punct = ' !' if greeting_fr else '.'
        sentence = sentence.strip('., ') + punct
        return sentence[0].upper() + sentence[1:]

    # ── Build verb phrase ─────────────────────────────────────────────────
    effective_subj = subj or 'moi'
    conj_table = _NEG_CONJ if negated else _CONJ
    verb_fr = conj_table.get(verb, {}).get(effective_subj, verb)

    # Clitic pronoun objects (me/te before verb for moi/tu direct objects)
    # Used for: attendre, aimer, comprendre, donner (indirect)
    _CLITIC = {'moi': ("m'", 'me '), 'tu': ("t'", 'te ')}
    clitic_pre = ''  # goes between subject and verb
    clitic_objs = [o for o in objects if o in ('moi', 'tu') and o != effective_subj]
    non_clitic_objs = [o for o in objects if o not in ('moi', 'tu') or o == effective_subj]
    if clitic_objs and verb in ('attendre', 'aimer', 'comprendre', 'donner'):
        cl = clitic_objs[0]
        vow_v, con_v = _CLITIC[cl]
        clitic_pre = vow_v if _vowel_start(verb_fr) else con_v

    # Subject pronoun: when negated the next word is 'ne', not the verb
    first_for_pron = (clitic_pre if clitic_pre else 'ne') if negated else (clitic_pre or verb_fr)
    subj_fr = _subject_fr(effective_subj, first_for_pron)
    # Fix elision: no space between j' and next word
    sep = '' if subj_fr.endswith("'") else ' '

    # ne/n' for negation
    if negated:
        neg_part = "n'" if _vowel_start(clitic_pre or verb_fr) else 'ne '
        vp = f"{subj_fr}{sep}{neg_part}{clitic_pre}{verb_fr}"
    else:
        vp = f"{subj_fr}{sep}{clitic_pre}{verb_fr}"

    # Object phrase (with prepositions)
    obj_parts = []
    _INDIRECT_VERBS = {'donner'}  # verbs that take à + recipient
    for i, o in enumerate(non_clitic_objs):
        o_fr = _OBJ_FR[o]
        # For donner: first obj is direct (aide/amour), second is indirect (à ami)
        if verb == 'donner' and i > 0:
            if o in ('ami',):
                o_fr = 'à ' + _OBJ_FR[o]
            elif o in ('bebe',):
                o_fr = 'au bébé'
            elif o in ('chat',):
                o_fr = 'au chat'
            elif o in ('moi',):
                o_fr = 'à moi'
            elif o in ('tu',):
                o_fr = 'à toi'
        obj_parts.append(o_fr)

    # Infinitive after modal verbs
    inf_str = ''
    if inf_verb and verb in ('vouloir', 'aimer', 'attendre') and not obj_parts:
        inf_str = inf_verb

    # avec
    avec_str = ''
    if avec_person:
        avec_str = 'avec ' + _AVEC_FR[avec_person]

    # After negation, "de l'" → "d'" (partitive after pas)
    if negated and obj_parts:
        obj_parts = [o.replace("de l'", "d'").replace("de l'", "d'") for o in obj_parts]

    # Assemble
    parts = [vp]
    if inf_str:
        parts.append(inf_str)
    if obj_parts:
        parts.append(' '.join(obj_parts))
    if avec_str:
        parts.append(avec_str)
    if mods:
        parts.extend(mods)

    body = ' '.join(parts)

    # ── Wrap in question form if needed ───────────────────────────────────
    if is_question:
        sentence = (temporal_prefix + 'Comment ' + body + ' ?').strip()
    else:
        punct = ' !' if greeting_fr else '.'
        sentence = (temporal_prefix or prefix) + body + punct

    # Capitalise
    sentence = sentence.strip()
    return sentence[0].upper() + sentence[1:] if sentence else ''


# ── Transformer model (primary neural translator, loaded lazily) ──────────────

_tf_translator = None

def _get_transformer():
    global _tf_translator
    if _tf_translator is None:
        try:
            from ml.transformer_model import TransformerTranslator
            m = TransformerTranslator()
            _tf_translator = m if m.load() else False
        except Exception:
            _tf_translator = False
    return _tf_translator if _tf_translator else None


# ── Neural slot model (optional, loaded lazily) ───────────────────────────

_SLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'slot_model')

class _SlotModel:
    """Wraps the BiLSTM multi-task slot-filler."""

    def __init__(self):
        self._model = None
        self._tok   = None
        self._cfg   = None

    def load(self) -> bool:
        model_path = os.path.join(_SLOT_DIR, 'model.keras')
        tok_path   = os.path.join(_SLOT_DIR, 'tokenizer.pkl')
        cfg_path   = os.path.join(_SLOT_DIR, 'config.json')
        if not all(os.path.exists(p) for p in (model_path, tok_path, cfg_path)):
            return False
        try:
            import json, pickle
            import numpy as np
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            self._model = load_model(model_path)
            with open(tok_path, 'rb') as f:
                self._tok = pickle.load(f)
            with open(cfg_path) as f:
                self._cfg = json.load(f)
            self._pad_sequences = pad_sequences
            self._np = np
            return True
        except Exception:
            return False

    def translate(self, signs_list: list) -> str:
        np  = self._np
        cfg = self._cfg
        raw = ' '.join(s.lower() for s in signs_list)
        seq = self._tok.texts_to_sequences([raw])
        seq = self._pad_sequences(seq, maxlen=cfg['max_len'], padding='post')
        preds = self._model.predict(seq, verbose=0)
        p_greet, p_subj, p_verb, p_inf, p_neg, p_obj, p_mods, p_avec = preds

        GREETINGS = cfg['greetings']
        SUBJECTS  = cfg['subjects']
        VERBS     = cfg['verbs']
        OBJECTS   = cfg['objects']
        MODS      = cfg['mods']
        AVEC      = cfg['avec']

        greet_str = GREETINGS[int(np.argmax(p_greet[0]))]
        subj_str  = SUBJECTS[int(np.argmax(p_subj[0]))]
        verb_str  = VERBS[int(np.argmax(p_verb[0]))]
        inf_str   = VERBS[int(np.argmax(p_inf[0]))]
        negated   = float(p_neg[0][0]) > 0.5
        objs      = [OBJECTS[i] for i, v in enumerate(p_obj[0]) if v > 0.5]
        mods      = [MODS[i]    for i, v in enumerate(p_mods[0]) if v > 0.5]
        avec_str  = AVEC[int(np.argmax(p_avec[0]))]

        reconstructed = []
        if greet_str != '<none>': reconstructed.append(greet_str)
        if subj_str  != '<none>': reconstructed.append(subj_str)
        if negated:               reconstructed.append('non')
        if verb_str  != '<none>': reconstructed.append(verb_str)
        if inf_str   != '<none>': reconstructed.append(inf_str)
        reconstructed.extend(objs)
        if avec_str  != '<none>': reconstructed.extend(['avec', avec_str])
        reconstructed.extend(mods)

        return rule_translate(reconstructed) if reconstructed else rule_translate(signs_list)


_slot_model: _SlotModel | None = None

def _get_slot_model() -> _SlotModel | None:
    global _slot_model
    if _slot_model is None:
        m = _SlotModel()
        _slot_model = m if m.load() else False
    return _slot_model if _slot_model else None


# ── Public API — drop-in replacement for the seq2seq Translator ────────────

class Translator:
    """
    Neuro-symbolic LSF→French translator.
    Uses BiLSTM slot-filler when available; falls back to deterministic rules.
    """

    def __init__(self):
        self.loaded = True  # always ready (rules are the fallback)
        self._transformer = None
        self._neural: _SlotModel | None = None

    def load(self):
        self._transformer = _get_transformer()
        self._neural      = _get_slot_model()

    def translate(self, signs_list: list) -> str:
        if self._transformer is not None:
            return self._transformer.translate(signs_list)
        if self._neural is not None:
            return self._neural.translate(signs_list)
        return rule_translate(signs_list)

    def score_signs(self, signs_list, max_steps=None) -> float:
        """Heuristic plausibility score: reward known subject+verb patterns."""
        if not signs_list:
            return 0.0
        signs = [s.lower() for s in signs_list]
        score = 0.0
        has_subj = any(s in _SUBJECTS for s in signs)
        has_verb = any(s in _VERBS for s in signs)
        has_dup  = len(signs) != len(set(signs))
        if has_subj:
            score += 1.0
        if has_verb:
            score += 1.0
        if has_subj and has_verb:
            score += 1.0
        if has_dup:
            score -= 1.0
        return score / max(1, len(signs_list))

    def prune_intruders(self, signs_list, min_gain=0.15, min_keep=1):
        return list(signs_list), []

    def reorder_signs(self, signs_list, min_gain=0.1, max_iter=40):
        return list(signs_list), False


# Singleton
_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
        _translator.load()
    return _translator
