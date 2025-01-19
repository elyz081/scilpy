#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script simply finds the 3 closest angular neighbors of each direction
(per shell) and compute the voxel-wise correlation.
If the angles or correlations to neighbors are below the shell average (by
args.std_scale x STD) it will flag the volume as a potential outlier.

This script supports multi-shells, but each shell is independant and detected
using the --b0_threshold parameter.

This script can be run before any processing to identify potential problem
before launching pre-processing.
"""

import argparse
import json
import logging

from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import applymask
import nibabel as nib
import numpy as np


from scilpy.dwi.operations import detect_volume_outliers
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              normalize_bvecs)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (assert_inputs_exist, add_b0_thresh_arg,
                             add_skip_b0_check_arg, add_verbose_arg,
                             add_json_args)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='The DWI file (.nii) to concatenate.')
    p.add_argument('in_bval',
                   help='The b-values files in FSL format (.bval).')
    p.add_argument('in_bvec',
                   help='The b-vectors files in FSL format (.bvec).')
    
    p.add_argument('--mask',
                   help='Optional mask file to exclude background '
                        'from correlation computation.')
    p.add_argument('--std_scale', type=float, default=2.0,
                   help='How many deviation from the mean are required to be '
                        'considered an outlier. [%(default)s]')
    p.add_argument('--out_file', type=str,
                   help='Optional output file (json) to dump the results.')



    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_json_args(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec],
                        optional=[args.mask])

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    data = nib.load(args.in_dwi).get_fdata()

    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        logging.debug(f'Loaded the masks with {np.count_nonzero(mask)} voxels')
        data = applymask(data, mask)
        logging.info("Mask applied.")

    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    bvecs = normalize_bvecs(bvecs)

    # If output_file not provided, not using the result.
    # Only printing on screen.
    results_dict, outliers_dict = detect_volume_outliers(data, bvals, bvecs, args.std_scale,
                                    args.b0_threshold)
    for key in results_dict:
        if isinstance(results_dict[key], np.ndarray):
            results_dict[key] = results_dict[key].tolist()
        elif isinstance(results_dict[key], dict):
            for subkey in results_dict[key]:
                if isinstance(results_dict[key][subkey], np.ndarray):
                    results_dict[key][subkey] = results_dict[key][subkey].tolist()

    for key in outliers_dict:
        if isinstance(outliers_dict[key], np.ndarray):
            outliers_dict[key] = outliers_dict[key].tolist()
        elif isinstance(outliers_dict[key], dict):
            for subkey in outliers_dict[key]:
                if isinstance(outliers_dict[key][subkey], np.ndarray):
                    outliers_dict[key][subkey] = outliers_dict[key][subkey].tolist()
    json_output = {
        'results': results_dict,
        'outliers': outliers_dict
    }

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(json_output, f, indent=args.indent, sort_keys=args.sort_keys)

if __name__ == "__main__":
    main()