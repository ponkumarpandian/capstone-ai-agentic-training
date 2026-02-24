"""
MediSuite Agent — CLI Entry Point
Run the claim processing workflow from the command line.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import settings
from orchestrator import MediSuiteOrchestrator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def print_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              🏥  MediSuite Agent System  🏥              ║")
    print("║         Multi-Agent Medical Claim Processing            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_results(results: dict) -> None:
    """Pretty-print the workflow results."""
    summary = results.get("summary", {})

    print()
    print("┌──────────────────────────────────────────────────────────┐")
    print("│                   WORKFLOW RESULTS                       │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│  Status      : {results.get('status', 'N/A'):<41}│")
    print(f"│  Duration    : {results.get('total_duration_s', 0):<41}│")
    print(f"│  Patient     : {summary.get('patient', 'N/A'):<41}│")
    print(f"│  Claim ID    : {summary.get('claim_id', 'N/A'):<41}│")
    print(f"│  Amount      : ${summary.get('amount', 0):,.2f}{'':<34}│")
    print(f"│  Decision    : {summary.get('decision', 'N/A'):<41}│")
    print(f"│  PDF         : {summary.get('pdf_path', 'N/A'):<41}│")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│  Justification: {summary.get('justification', 'N/A')[:39]:<39}│")
    print("└──────────────────────────────────────────────────────────┘")

    # Agent step details
    steps = results.get("steps", {})
    for step_name, step_data in steps.items():
        if isinstance(step_data, dict) and "error" not in step_data:
            duration = step_data.get("duration_s", "?")
            print(f"\n  ✔ {step_name:<25} ({duration}s)")
        elif isinstance(step_data, dict):
            print(f"\n  ✘ {step_name:<25} ERROR: {step_data.get('error', 'Unknown')}")

    # Errors
    errors = results.get("errors", [])
    if errors:
        print("\n⚠ Errors:")
        for err in errors:
            print(f"  - {err}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediSuite Agent — Medical Claim Processing CLI",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=str(settings.data_dir / "sample_patient.json"),
        help="Path to patient information JSON file",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=str(settings.data_dir / "sample_clinical_notes.txt"),
        help="Path to clinical notes text file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(settings.data_dir / "sample_document_metadata.json"),
        help="Path to document metadata JSON file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full results as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate inputs
    for label, path in [("Patient", args.patient), ("Notes", args.notes)]:
        if not Path(path).exists():
            print(f"❌ {label} file not found: {path}")
            sys.exit(1)

    print_banner()
    print(f"  Patient file : {args.patient}")
    print(f"  Notes file   : {args.notes}")
    print(f"  Metadata     : {args.metadata}")
    print()

    # Run workflow
    orchestrator = MediSuiteOrchestrator()
    results = orchestrator.run_workflow(
        patient_data_path=args.patient,
        clinical_notes_path=args.notes,
        document_metadata_path=args.metadata if Path(args.metadata).exists() else None,
    )

    print_results(results)

    # Save full JSON output
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  📄 Full results saved to: {out_path}")
        print()


if __name__ == "__main__":
    main()
