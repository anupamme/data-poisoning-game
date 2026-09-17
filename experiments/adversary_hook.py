"""The adversarial update-space step, in ONE place for every dose runner.

Two runners apply it -- experiments/run_dose_response.py (the confounded ladder) and
experiments/run_targeted_dose.py (the Mode S/A/M instrument) -- and they must apply the
IDENTICAL step or the design comparison they exist to support is comparing two adversaries
instead of two designs. It lives here rather than in each runner for the reason
run_targeted_dose's own docstring gives about run_one: a copy is how two suites drift apart
in what they compute.

Two adversary paths, MUTUALLY EXCLUSIVE:

  ca is None            per-client atk.manipulate_update -- the frozen behaviour of both
                        runners, unchanged.
  ca = (eps, decorr)    criterion_aware_updates SUBSTITUTES for manipulate_update. This
                        mirrors run_criterion_aware_adversary.py:218-225, where the plain
                        backdoor and the criterion-aware construction are an if/elif and
                        never compose. Adversaries train on poisoned data in BOTH cases;
                        only this update-space step differs.

WHY THE VERIFY PATH IS NOT "assert the tensor changed". For `backdoor_pixel`,
manipulate_update is the inherited identity (attacks/attack_strategies.py:61-63) -- the
backdoor lives entirely in poison_dataset -- so "changed" is false by design. Only
`model_scaling` overrides it (:137, a x10 elementwise multiply). An assertion that demanded
change would fail on the correct configuration, and one that merely demanded "something
happened" would not have caught the defect it exists for: a cross-distribution arm in this
repo once came out bit-identical to another because manipulate_update was never called, and
since pixel's hook is a no-op anyway the two arms silently collapsed onto one computation.
So verify() pins the EXPECTED behaviour per attack -- change iff the attack overrides the
base method -- which catches both a hook that never fires and an attack object that is not
the one the arm thinks it is.
"""
import torch

from attacks.attack_strategies import AttackStrategy


def _criterion_aware_updates():
    """Imported LAZILY, and that is load-bearing rather than a style choice.

    run_criterion_aware_adversary.py runs os.makedirs(results/criterion_aware_adversary) at
    MODULE level (:75). Both frozen dose runners import this module, so a top-level import
    here would make merely importing run_dose_response touch a results/ directory -- exactly
    what that runner's own comment at :145 forbids, because it makes the provenance gate
    ("prereg committed before the first write to the output directory") ambiguous. Deferring
    the import means the ca=None path -- every existing call site -- never touches it.
    """
    from experiments.run_criterion_aware_adversary import criterion_aware_updates
    return criterion_aware_updates


def defines_own_manipulate(atk):
    """True iff this attack overrides the base identity manipulate_update."""
    return type(atk).manipulate_update is not AttackStrategy.manipulate_update


def apply_adversary(ups, pids, adv, atk, global_model, ca=None, verify=False):
    """Replace the adversarial entries of `ups` in participation order. Returns the list.

    ups          list of update dicts, one per participant, in participation order
    pids         participant client ids, same order as ups
    adv          set of adversarial client ids
    ca           None, or (eps, decorrelate) for the criterion-aware construction
    verify       raise unless exactly one path ran and it behaved as the attack defines
    """
    adv_local = [i for i, cid in enumerate(pids) if cid in adv]
    if not adv_local:
        return ups

    probe = adv_local[0]
    before = {k: v.detach().clone() for k, v in ups[probe].items()} if verify else None

    if ca is None:
        for i in adv_local:
            ups[i] = atk.manipulate_update(ups[i], global_model)
        path = "manipulate_update"
    else:
        eps, decorrelate = ca
        benign_local = [i for i, cid in enumerate(pids) if cid not in adv]
        ups = _criterion_aware_updates()(ups, adv_local, benign_local,
                                         list(ups[0].keys()), eps, decorrelate)
        path = "criterion_aware_updates"

    if verify:
        after = ups[probe]
        changed = any(not torch.equal(before[k], after[k].detach()) for k in before)
        if ca is not None:
            expect, why = True, ("criterion_aware_updates must rewrite the adversarial "
                                 "update; it did not")
        elif defines_own_manipulate(atk):
            expect, why = True, (f"{type(atk).__name__} overrides manipulate_update, so the "
                                 f"adversarial update must change; it did not -- the hook "
                                 f"did not fire")
        else:
            expect, why = False, (f"{type(atk).__name__} inherits the identity "
                                  f"manipulate_update, so the adversarial update must be "
                                  f"unchanged; it changed -- this is not the attack the arm "
                                  f"thinks it is")
        if changed is not expect:
            raise AssertionError(f"[adversary_hook] {path}: {why}")

    return ups
