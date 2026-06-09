"""Smoke-check the Condor connection + SciToken. Prints schedd identity + a one-row query."""

import os
import sys


def main():
    # Local import keeps the script useful even when the rest of the plugin can't import cleanly.
    import htcondor

    if len(sys.argv) > 1:
        os.environ["BEARER_TOKEN_FILE"] = os.path.expanduser(sys.argv[1])
    if "BEARER_TOKEN_FILE" in os.environ:
        print(f"using SciToken at {os.environ['BEARER_TOKEN_FILE']}")
    else:
        print("note: no BEARER_TOKEN_FILE set; htcondor will use whatever it can find")

    schedd = htcondor.Schedd()
    print(f"schedd located: {schedd}")
    ads = schedd.query(
        constraint="True",
        projection=["ClusterId", "Owner", "JobStatus"],
        limit=1,
    )
    print(f"sample query returned {len(ads)} ad(s)")
    if ads:
        print(f"first ad: {dict(ads[0])}")


if __name__ == "__main__":
    main()
