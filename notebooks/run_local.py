"""Run every analysis stage on a local machine, in order, with all CPU cores.

Same functions as the Colab notebooks (01-07); use this when a long unattended run is easier locally.
Usage:  python run_local.py <project-folder-on-Google-Drive> [first_stage] [last_stage]
Stages: 1 download, 2 build, 3 pilot, 4 main, 5 hindcast, 6 sensitivity, 7 results.
Every stage is resumable: finished downloads and checkpoint parts on disk are skipped.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    import pandas as pd

    import p1fig
    import p1lib as L
    import p1run as R

    root = sys.argv[1]
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    P = L.Paths(root)

    def stage(n, name, fn):
        if first <= n <= last:
            L.log(f"=== stage {n}: {name} ===")
            R.record_environment(P, f"run_local stage {n} {name}")
            t = time.time()
            out = fn()
            L.log(f"=== stage {n} done in {(time.time() - t) / 60:.1f} min ===")
            return out

    stage(1, "download", lambda: R.nb01_download(P, threads=16))
    stage(2, "build", lambda: print(R.nb02_build(P, threads=16).to_string()))
    stage(3, "pilot", lambda: print(R.nb03_pilot(P)[0].to_string()))
    stage(4, "main", lambda: [R.nb04_main(P, pollutant=p) for p in L.POLLUTANTS])
    stage(5, "hindcast", lambda: [R.nb05_hindcast(P, pollutant=p) for p in L.POLLUTANTS])
    stage(6, "sensitivity", lambda: (R.nb06_sensitivity(P), R.nb06b_mc_error(P)))

    def results():
        main_ = pd.concat([pd.read_csv(f) for f in sorted(P.outputs.glob("main_*_station_results.csv"))], ignore_index=True)
        rows = pd.concat([pd.read_csv(f) for f in sorted(P.outputs.glob("hindcast_*_rows.csv.gz"))], ignore_index=True)
        scores, coverage = R.score_tables(P, rows)
        print(R.success_check(P, rows, coverage).to_string())
        print(R.summaries(P, main_).to_string())
        print(R.sensitivity_summary(P, main_).to_string())
        p1fig.make_all(P, main_, scores)

    stage(7, "results", results)


if __name__ == "__main__":  # required for multiprocessing on Windows
    main()
