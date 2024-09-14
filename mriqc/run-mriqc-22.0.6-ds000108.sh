#!/bin/bash

# Author: Yann Leprince, 2021-2023. <yann.leprince@cea.fr>

die () {
    echo "Aborting."
    exit 1
}

echo_and_run () {
    echo "\$ $*"
    if [ -z "$DRY_RUN" ]; then
        "$@" 2>&1 >> "$OUTPUT_DIR"/mriqc_run.log
    fi
}

glob_matches () {
    if [ $# -eq 1 ]; then
        [ -e "$1" ]
    else
        return 0
    fi
}


MRIQC_VERSION=22.0.6
MRIQC_CONTAINER=/i2bm/local/bids-apps/mriqc_${MRIQC_VERSION}.sif

RAWDATA=/neurospin/nfbd/Decoding/ds000108/rawdata
OUTPUT_DIR=/neurospin/nfbd/Decoding/ds000108/derivatives/mriqc-${MRIQC_VERSION}

DRY_RUN=

cd "$RAWDATA" || die

mkdir -p -m 1777 /volatile/tmp
mkdir -p "${OUTPUT_DIR}"

for sub_dir in sub-*/; do
    sub_entity=${sub_dir%/}
    sub_label=${sub_entity#sub-}
    pushd "$sub_dir" >/dev/null || die
    if glob_matches "$OUTPUT_DIR/${sub_entity}/anat/${sub_entity}"*.json; then
        echo "Skipping ${sub_entity}: already processed"
    else
        # FIXME: mriqc (as of version 0.16.1) leaves behind a ton of Xvfb
        # processes, which keep the container running and end up using up
        # all the available loop devices, preventing the launch of further
        # containers...
        tmpdir=$(mktemp -d --tmpdir=/volatile/tmp mriqc.XXXXXXXXXX)
        echo_and_run timeout --kill-after=1m 2h singularity run \
                     --cleanenv \
                     --bind "$RAWDATA":"$RAWDATA":ro \
                     --bind "$OUTPUT_DIR":/out \
                     --bind "$tmpdir":/tmpdir \
                     "${MRIQC_CONTAINER}" \
                     --work-dir /tmpdir \
                     "$RAWDATA"\
                     /out \
                     participant \
                     --participant-label "${sub_label}"
        # Clean up temporary files that mriqc leaves behind
        rm -rf "$tmpdir"
    fi
    popd >/dev/null || die
done

tmpdir=$(mktemp -d --tmpdir=/volatile/tmp mriqc.XXXXXXXXXX)
echo_and_run singularity run \
    --cleanenv \
    --bind "$RAWDATA":"$RAWDATA":ro \
    --bind "$OUTPUT_DIR":/out \
    --bind "$tmpdir":/tmpdir \
    "${MRIQC_CONTAINER}" \
    --work-dir /tmpdir \
    "$RAWDATA" \
    /out \
    group
rm -rf "$tmpdir"
