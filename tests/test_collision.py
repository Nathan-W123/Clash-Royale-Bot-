"""Ground-body collision: proves funneling/shelter, not just no-overlap.

Two expectations here were relaxed by the obstacle steering added in #36
(see docs/SIM_FIDELITY.md). Bodies still cannot overlap and cannot be walked
through, but a single body no longer walls off a lane forever — units route
around it, which is what the real game does. Shelter is therefore asserted as
a measurable *delay* rather than as impassability.

Several tests manually force `target_id` to isolate collision from the
engine's own nearest-enemy targeting choice. Where a "wall" needs to sit in
the mover's path without ever becoming its target (so it isn't exempted from
collision as "the thing I'm attacking"), the wall is spawned on the *mover's
own side* — the targeting AI only ever considers enemies, so a friendly body
can never be acquired as a target, while collision still applies to it
regardless of side.
"""
import pytest

from src.simulator.constants import Side
from src.simulator.targeting import dist
from tests.conftest import dummy_stats, make_engine, spawn_unit


def test_a_body_in_the_way_diverts_rather_than_walls_off(cards, arena):
    """A unit can't walk *through* a body — but it does walk *around* one.

    This expectation changed with the obstacle steering added in #36. The
    previous model had the mover stall against the wall and stay there for
    the rest of the match, because its waypoint never changed so it
    re-proposed the same blocked step forever. Real units flow around each
    other; a single body costs a detour, not the lane. What collision still
    guarantees — and what this asserts — is that the two bodies never
    overlap and that the mover is pushed off its straight line.
    """
    eng = make_engine(cards, arena)
    mover = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 25.0)
    # Friendly to mover, so targeting AI (enemies-only) can never acquire it —
    # it can only ever affect the mover by physically blocking, not by being
    # picked as a target (which would exempt it from collision).
    wall = spawn_unit(eng, dummy_stats("wall", speed=0.0), Side.TOP, 9.0, 22.0)
    far_target = spawn_unit(eng, dummy_stats("far", speed=0.0), Side.BOTTOM, 9.0, 18.0)
    mover.target_id = far_target.id  # within lock-break range (7 <= 8.25); never re-acquires

    detoured = False
    for _ in range(150):  # 15s — plenty of time to reach far_target if unobstructed
        eng.tick()
        assert dist(mover.x, mover.y, wall.x, wall.y) >= \
            mover.radius + wall.radius - 1e-6, "never passes through the body"
        detoured = detoured or abs(mover.x - 9.0) > 0.5

    assert detoured, "steering must push it off the blocked straight line"
    assert far_target.hp < far_target.stats.hp, "and it does eventually get there"


def test_a_wall_of_bodies_blocks_the_lane_outright(cards, arena):
    """Steering around *one* body is correct; steering around five is not.

    The complement to `test_a_body_in_the_way_diverts_rather_than_walls_off`:
    if every sidestep angle is also occupied there is nowhere to flow to, and
    the lane is genuinely shut. Without this, "collision" would only ever be
    a speed bump and the funnelling players build on purpose would not exist.
    Measured behaviour: no wall reaches in ~30 ticks, one blocker ~36, three
    or more never arrive at all.
    """
    eng = make_engine(cards, arena)
    mover = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 25.0)
    for x in (7.0, 8.0, 9.0, 10.0, 11.0):
        # Friendly to the mover so they can only ever block, never be targeted.
        spawn_unit(eng, dummy_stats(f"wall_{x}", speed=0.0), Side.TOP, x, 22.0)
    far_target = spawn_unit(eng, dummy_stats("far", speed=0.0), Side.BOTTOM, 9.0, 18.0)
    mover.target_id = far_target.id

    for _ in range(200):  # 20s, far longer than an unobstructed approach needs
        eng.tick()

    assert far_target.hp == far_target.stats.hp, "wall must not be passable"
    assert mover.y > 21.0, "mover should still be stuck on the far side"


def test_collision_prevents_any_overlap_during_a_converging_swarm(cards, arena):
    """Several units converging on the same point funnel around it rather
    than stacking exactly on top of each other or of the target."""
    eng = make_engine(cards, arena)
    # Both below the river (no bridge detour) and far enough from every tower
    # that one doesn't snipe a skeleton mid-test (towers gate on edge
    # distance, i.e. center distance minus both radii — worth remembering
    # when placing test units near arena edges).
    target = spawn_unit(eng, dummy_stats("bullseye", speed=0.0, hp=1e9), Side.TOP, 9.0, 14.5)
    movers = [
        spawn_unit(eng, cards["skeletons"], Side.BOTTOM, x, 8.5)
        for x in (7.0, 8.0, 9.0, 10.0, 11.0)
    ]
    start_dists = {m.id: dist(m.x, m.y, target.x, target.y) for m in movers}
    for m in movers:
        m.target_id = target.id

    for _ in range(200):  # 20s
        for m in movers:
            if m.hp > 0:
                m.target_id = target.id  # keep them converging on the same point
        eng.tick()
        alive = [m for m in movers if m.hp > 0]
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                assert dist(a.x, a.y, b.x, b.y) >= a.radius + b.radius - 1e-6
            assert dist(a.x, a.y, target.x, target.y) >= a.radius + target.radius - 1e-6

    # They actually converged (funneled in) rather than stalling far from the target.
    for m in movers:
        assert dist(m.x, m.y, target.x, target.y) < start_dists[m.id] * 0.75


def test_a_tank_in_the_path_buys_time_for_what_is_behind_it(cards, arena):
    """Shelter is a *delay*, measured against the same fight without the tank.

    Also changed by #36's steering: a body no longer makes what is behind it
    permanently unreachable, which was never how the real game behaves. What
    it does — and what makes tanking worth elixir — is cost the attacker the
    time it takes to walk around.
    """

    def time_to_kill(with_tank: bool) -> int:
        eng = make_engine(cards, arena)
        # All below the river (y < 15, no bridge detour). The attacker ends up
        # within a princess tower's edge-distance range (towers are big —
        # radius 1.0 — and reach further than the raw range number suggests
        # once you subtract body radii), so it's given enough HP that a few
        # incidental tower hits can't confound the result.
        if with_tank:
            spawn_unit(eng, dummy_stats("tank", speed=0.0), Side.BOTTOM, 9.0, 11.2)
        protected = spawn_unit(eng, dummy_stats("vip", speed=0.0, hp=200.0),
                               Side.BOTTOM, 9.0, 8.0)
        attacker = spawn_unit(eng, dummy_stats(
            "attacker", hp=5000.0, damage=290.0, hit_speed=1.8, range=1.2, speed=1.25),
            Side.TOP, 9.0, 14.0)
        attacker.target_id = protected.id
        for tick in range(300):
            eng.tick()
            if protected.hp <= 0:
                return tick
        return 300

    sheltered = time_to_kill(with_tank=True)
    exposed = time_to_kill(with_tank=False)
    assert sheltered > exposed, "the tank must cost the attacker real time"


def test_flying_units_are_not_subject_to_ground_collision(cards, arena):
    eng = make_engine(cards, arena)
    flyer = spawn_unit(eng, cards["minions"], Side.TOP, 9.0, 20.0)
    # Sits directly on the straight-line path between flyer and target.
    ground_wall = spawn_unit(eng, dummy_stats("wall", speed=0.0), Side.TOP, 9.0, 15.0)
    # Within the 8.25-tile lock-break radius so the forced target isn't
    # dropped for a distant tower before the flyer gets moving.
    far_target = spawn_unit(eng, dummy_stats("far", speed=0.0, hp=1.0), Side.BOTTOM, 9.0, 13.0)
    flyer.target_id = far_target.id

    for _ in range(150):
        eng.tick()
        if far_target.hp <= 0:
            break
    assert far_target.hp <= 0  # flew straight past the ground wall unobstructed


def test_unit_can_still_close_the_final_distance_onto_its_own_target(cards, arena):
    """Collision must never block a unit from reaching what it's attacking —
    only from walking through *other* bodies."""
    eng = make_engine(cards, arena)
    knight = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 13.5)
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.4)
    knight.target_id = bag.id
    for _ in range(50):
        eng.tick()
    assert bag.hp < bag.stats.hp  # engaged and landed hits normally
