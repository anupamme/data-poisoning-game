# Quotation-integrity repair log (Round 57, literature audit)

Every quotation in every coding record is now asserted VERBATIM against the paper's extracted
full text by `experiments/verify_quotes_literature_audit.py`, which exits non-zero on any failure.
This file records what that check found the first time it was run over the whole set, and what was
done about it. It exists because non-negotiable 5 of the frozen pre-registration says nothing is
transcribed by hand, and the check is what makes that claim auditable rather than asserted.

**Scale.** 37 records, 185 quotations, 197 spans. The first run failed 28 quotations across 15
records. One further quotation (`2508.12978` C-c) had already failed the same assertion while its
record was being written and was repaired before the record shipped, so **26 quotations in 16
records needed repair** and 159 were verbatim as written.

**No code changed.** Every repair was to the transcription of evidence, not to an adjudication:
all five criteria in all 16 affected records keep the code they had. Two repairs produced STRONGER
evidence for the code already recorded (`2502.00587` C-b, `2505.10297` C-e). That is a real finding
and not a reassuring one: it means the defects were of a kind that a reader checking the argument
would not have caught, because the paraphrases were faithful to the sense. The only reason it can be
stated at all is that the quotations are now machine-checked.

**How the repairs were made.** No quotation was retyped. Each replacement was sliced out of the
extracted full text by anchor, asserted to be a substring of it, and written back programmatically;
backups of the pre-repair records are in `results/literature_audit/backup_pre_quote_repair/`.

## The ten classes, worst first

| paper | criterion | class | what was wrong |
|---|---|---|---|
| `2502.00587` | C-b | invented completion, conflation | THE MOST SERIOUS CLASS, AND IT SUPPORTED A C-b = YES. The bracket read "an[omalous models may remain, which]" and the tail read "clustering but still deviate from the central tendency of the benign models". The paper's actual sentence is "some malicious or anomalous updates may still be present", and the phrase "central tendency" does not continue it; the true continuation names the median as what handles the residue. Replaced by the source passage, which is STRONGER evidence for the same code. |
| `2502.00587` | C-e | invented completion, apostrophe | The bracket "an[omalous models remain]" was likewise wrong ("anomalous updates may still be present"), and "RKD’s" was typed "RKD's". |
| `2502.07011` | C-b | invented completion | THIS ONE SITS IN A PAPER THAT CARRIES A PRIMARY CODE. The bracket "is [below a threshold]" was coder paraphrase; the paper says "which assume that the number of malicious updates in a round is smaller than the number of benign ones". Same claim, different words -- and the words are the evidence. |
| `2503.18284` | C-d | invented completion, symbol | The displayed fraction 1/K, which pdftotext emits out of order as "the weighting 1 . factors are set as K", had been healed to "the weighting factors are set as [1/K]"; and the citation range "[27]–[29]" had been typed with an ASCII hyphen. Now an explicit elision that stops before the fraction. |
| `2306.12608` | C-b | conflation | THE MOST SERIOUS CLASS. The quotation opened with an invented connective, "Taken together,", which occurs nowhere in the paper, and then fused the abstract's claim ("our proposed protocols achieve better privacy-utility tradeoff and stronger Byzantine robustness than several baseline methods") with the conclusion's ("the advantage of DP-BREM/+ ... over five baseline protocols"). Replaced by the conclusion sentence alone. |
| `2503.18284` | C-b | column splice | The extracted stream interleaves the two columns of an IEEE two-column layout, so the sentence "However, low-precision quantization compromises performance and majority voting is inadequate in withstanding formidable Byzantine attacks" is broken up by text from the adjacent column. The quotation had spliced the adjacent column's word "clustering" into it, producing "low-precision clustering [quantization compromises...]" -- a sentence the paper does not contain. Now quoted as two verbatim spans. |
| `2404.04139` | C-a | head rewrite, ligature | The opening was rewritten from the paper's "we present FedZZ, which harnesses a zone-based deviating update (ZBDU) mechanism" to "We propose a zone-based deviating update (ZBDU) mechanism", which attributes to the paper a sentence it does not contain; and two ﬀ/ﬁ ligatures were expanded. |
| `2404.04139` | C-b | head rewrite | The bullet's opening "We design and implement FedZZ, which uses" was rewritten as "we introduce FedZZ, which uses". |
| `2207.01982` | C-b | unmarked elision | The word "parameter" was dropped from "the parameter gradients corresponding to the neurons" with no elision marker. |
| `2502.05547` | C-d | layout interruption | A page number, extracted as a bare "5" between two sentences, had been silently deleted. The quotation is now an explicit two-span elision. |
| `2505.10297` | C-e | layout interruption | An entire table (TABLE 4's caption and body) falls between the two halves of this sentence in the extracted stream, and had been silently closed up. Now an explicit two-span elision, and the second span was extended to include the sentence about complementarity, which strengthens the code. |
| `1909.05125` | C-a | case, apostrophe | Sentence-initial "We" lowercased; two U+2019 apostrophes typed as "'". |
| `2505.10297` | C-c | case | The table caption is set in small caps and extracts as "R ESULTS REPORTED AS MA(%) / BA(%)"; it had been re-cased to "Results reported as MA(%) / BA(%)." Now quoted as extracted, with the artifact named in the locator. |
| `2104.06685` | C-d | symbol | The threshold symbol δ typed as the word "delta". |
| `2206.12100` | C-c | symbol | A true minus U+2212 in "0.5 − 2.8%" typed as an ASCII hyphen. |
| `2208.10161` | C-a | symbol | "sign(ĝit ) ∈ {−1, +1}" typed as "sign(g_i^t) in {-1, +1}" -- a set-membership sign, a hatted subscripted variable and a true minus all transliterated. |
| `2210.01437` | C-c | symbol | The figure caption's α typed as the word "alpha". |
| `2210.01437` | C-d | symbol | Same caption, same substitution. |
| `2404.04139` | C-d | ligature, symbol | "Conﬁgurations"/"conﬁgurations"/"diﬀerent"/"Speciﬁcally" ligatures expanded and the math-italic 𝑚 typed as "m". |
| `2509.08089` | C-d | apostrophe, symbol | "CSFT’s" typed as "CSFT's" and the math-italic 𝛿 typed as Greek δ. |
| `2509.08089` | C-e | apostrophe, symbol | Same two substitutions. |
| `2404.04139` | C-c | ligature | "eﬀectively" typed as "effectively". |
| `2207.01982` | C-a | apostrophe | "peers’" typed as "peers'". |
| `2306.12608` | C-a | apostrophe | "clients’" typed as "clients'". |
| `2306.12608` | C-e | apostrophe | "client’s" typed as "client's". The bracketed completion "[are more Byzantine clients]" was CORRECT and is now inlined and verified. |
| `2310.13403` | C-e | apostrophe | "nodes’" typed as "nodes'". |

## Why the checker missed four of these on its first design

The first version tested only the text BEFORE a bracketed completion, on the reasoning that the
bracket marks coder-supplied text as such. Four invented completions passed it, including one in
`2502.07011`, a paper that carries a primary code, and one in `2502.00587` whose invented tail was
contradicted by the paper's own sentence. The check now requires the completion to be verbatim too,
with numeric citation markers masked so their brackets are not mistaken for completions.

It also had no notion of elision, so a legitimate " ... " quotation failed as a whole. Spans are now
checked separately AND in source order, which surfaced a pair in `2306.04984` quoted
problem-then-solution from two different sections; that locator now says so.

## What this means for the LaTeX table in App. G

Verbatim quotations now contain characters the paper's own punctuation gate forbids in source:
U+2212 (`2206.12100`), U+2013 in a citation range (`2503.18284`), U+2019, the ﬀ/ﬁ ligatures, Greek
and math-italic letters, and a set-membership sign. The audit table therefore cannot paste these
spans as bytes. The substitution used to render them must be listed in the appendix alongside the
table, and the machine-checked JSON records remain the authority for what each paper actually says.
