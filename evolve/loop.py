"""Luna Evolve main loop CLI.

Usage:
  python -m evolve.loop --preset tiny --generations 3 --population 4 --train_steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

from evolve.archive import ParetoArchive
from evolve.fitness import evaluate_genome
from evolve.genome import Genome, crossover, mutate, seed_population


def _genome_intent_text(genome: Genome) -> str:
    """One-line description of a genome's direction for SafetyCTM think-before-mutate."""
    a = genome.arch
    return (
        f"evolve preset={genome.preset} mamba_ratio={a.mamba_ratio:.2f} "
        f"experts={a.num_routed_experts} top_k={a.top_k} "
        f"ctm_ticks={a.ctm_max_ticks} inject_every={a.ctm_inject_every} "
        f"kv_lora={a.kv_lora_rank} intermediate={a.intermediate_size}"
    )


def run_evolution(args: argparse.Namespace) -> ParetoArchive:
    if args.preset in ("550b", "77b_active") and not args.allow_large:
        raise SystemExit(
            "Preset 550b/77b_active blocked. Use --allow-large only after smaller ladders stabilize."
        )

    rng = random.Random(args.seed)
    archive = ParetoArchive(max_size=args.archive_size)
    population: List[Genome] = seed_population(args.preset, args.population, rng)

    # CTM cognitive judge for think-before-mutate. The charter hard floor
    # already applies via the fitness safety gate; this is an additional soft
    # filter that skips genomes whose direction the CTM judges unsafe.
    safety_ctm = None
    if not getattr(args, "no_ctm", False):
        from safety.cognition import default_safety_ctm
        safety_ctm = default_safety_ctm()

    os.makedirs(args.output_dir, exist_ok=True)

    for gen in range(args.generations):
        print(f"\n=== Generation {gen} | pop={len(population)} ===")
        evaluated = []
        for g in population:
            g.generation = gen
            # Think-before-mutate: CTM judges the genome's intent; refused
            # genomes are skipped and never sent to short-training.
            if safety_ctm is not None:
                j = safety_ctm.judge(_genome_intent_text(g))
                if not j.allow:
                    print(f"  {g.genome_id}: ctm_refused "
                          f"(sim={j.max_forbidden_sim:.3f} conf={j.confidence:.3f}) — skipped")
                    continue
            ind = evaluate_genome(
                g,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                allow_large=args.allow_large,
            )
            accepted = archive.add(ind)
            evaluated.append(ind)
            print(
                f"  {g.genome_id}: quality={ind.quality:.4f} "
                f"ce={ind.ce_loss:.3f} ticks={ind.avg_ticks:.2f} "
                f"FLOPs={ind.active_flops:.2e} VRAM~{ind.peak_vram_gb:.3f}GB "
                f"score={ind.scalar_score():.4f} archive={'Y' if accepted else 'n'}"
            )

        # Next population: champion + mutants + crossovers
        champion = archive.champion()
        next_pop: List[Genome] = []
        if champion is not None:
            next_pop.append(champion.genome.clone())

        ranked = sorted(evaluated, key=lambda x: x.scalar_score(), reverse=True)
        parents = [r.genome for r in ranked[: max(2, args.population // 2)]]

        while len(next_pop) < args.population:
            if rng.random() < 0.35 and len(parents) >= 2:
                a, b = rng.sample(parents, 2)
                child = crossover(a, b, rng)
            else:
                child = mutate(rng.choice(parents), rng)
            next_pop.append(child)

        population = next_pop[: args.population]

        # Persist generation snapshot
        snap = {
            "generation": gen,
            "archive": [m.to_dict() for m in archive.members],
            "champion": champion.to_dict() if champion else None,
        }
        with open(os.path.join(args.output_dir, f"gen_{gen}.json"), "w") as f:
            json.dump(snap, f, indent=2)

    archive.save(os.path.join(args.output_dir, "pareto_archive.json"))
    champ = archive.champion()
    if champ is not None:
        with open(os.path.join(args.output_dir, "champion.json"), "w") as f:
            json.dump(champ.to_dict(), f, indent=2)
        print(
            f"\nChampion {champ.genome.genome_id}: "
            f"quality={champ.quality:.4f} score={champ.scalar_score():.4f}"
        )
    print(f"Archive size: {len(archive)} → {args.output_dir}")
    return archive


def main():
    p = argparse.ArgumentParser(description="Luna Evolve outer loop")
    p.add_argument("--preset", type=str, default="tiny")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--train_steps", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--archive_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./evolve_runs")
    p.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow 550b/77b_active materialization (expensive)",
    )
    p.add_argument(
        "--no-ctm",
        action="store_true",
        help="Disable the CTM think-before-mutate judge (safety gate still applies)",
    )
    args = p.parse_args()
    run_evolution(args)


if __name__ == "__main__":
    main()
