#!/bin/bash

# Author: Pamela Pindi - 27.03.2023

die () {
	echo "Aborting."
	exit 1
}

glob_matches () {
	if [ $# -eq 1 ]; then
		[ -e "$1" ]
	else
		return 0
	fi
}

RAWDATA=/neurospin/nfbd/Decoding/ds000108/rawdata
OUTPUT_DIR=/neurospin/nfbd/Decoding/ds000108/derivatives/fmriprep-22.1.1

cd "$RAWDATA" || die

mkdir -p -m 1777 /volatile/tmp/fmriprep
mkdir -p "${OUTPUT_DIR}"

for sub_dir in sub-*/; do
    sub_entity=${sub_dir%/}
    sub_label=${sub_entity#sub-}
    pushd "$sub_dir" >/dev/null || die
	if glob_matches "$OUTPUT_DIR/${sub_entity}/ses-post/func/${sub_entity}_desc-brain_mask".json; then
		echo "Skipping ${sub_entity}: already processed"
	else 
		tmpdir=$(mktemp -d --tmpdir=/volatile/tmp/fmriprep)
		singularity run \
			--cleanenv \
			--bind /i2bm/local/freesurfer/license.txt:/freesurfer-license.txt:ro \
			--bind "$RAWDATA":"$RAWDATA":ro \
                	--bind "$OUTPUT_DIR":/out \
                	--bind "$tmpdir":/tmpdir:rw \
			/i2bm/local/bids-apps/fmriprep_22.1.1.sif --skip_bids_validation \
			--work-dir=/tmpdir --clean-workdir --fs-license-file=/freesurfer-license.txt --fs-no-reconall \
			"$RAWDATA" \
			/out \
			participant \
			--participant-label "${sub_label}"
		rm -rf "$tmpdir"

	fi
	popd >/dev/null || die
done



