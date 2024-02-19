#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on data per point (dpp) from input streamlines.

Although the input data always comes from the dpp, the output can be either
a dpp or a data_per_streamline (dps), depending on the chosen options.
Two modes of operation are supported: dpp and dps.
   - In dps mode, the operation is performed on dpp across the dimension of
   the streamlines resulting in a single value (or array in the 4D case)
   per streamline, stored as dps.
   - In dpp mode, the operation is performed on each point separately, resulting in
   a new dpp.

If endpoints_only and dpp mode is set the operation will only
be calculated at the streamline endpoints the rest of the
values along the streamline will be NaN

If endpoints_only and dps mode is set operation will be calculated
across the data at the endpoints and stored as a
single value (or array in the 4D case) per streamline.

Endpoint only operation:
correlation: correlation calculated between arrays extracted from
streamline endpoints (data must be multivalued per point) and dps
mode must be set.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram, StatefulTractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_operations import (
        perform_pairwise_streamline_operation_on_endpoints,
        perform_streamline_operation_per_point,
        perform_operation_per_streamline)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', metavar='OPERATION',
                   choices=['mean', 'sum', 'min',
                            'max', 'correlation'],
                   help='The type of operation to be performed on the \n'
                        'streamlines. Must\nbe one of the following: \n'
                        '%(choices)s.')
    p.add_argument('in_tractogram', metavar='INPUT_FILE',
                   help='Input tractogram containing streamlines and \n'
                        'metadata.')
    p.add_argument('--mode', required=True, choices=['dpp', 'dps'],
                   help='Set to dps if the operation is to be performed \n'
                   'across all dimensions resulting in a single value per \n'
                   'streamline. Set to dpp if the operation is to be \n'
                   'performed on each point separately resulting in a single \n'
                   'value per point.')
    p.add_argument('--in_dpp_name',  nargs='+', required=True,
                   help='Name or list of names of the data_per_point for \n'
                        'operation to be performed on. If more than one dpp \n'
                        'is selected, the same operation will be applied \n'
                        'separately to each one.')
    p.add_argument('--out_name', nargs='+', required=True,
                   help='Name of the resulting data_per_point or \n'
                   'data_per_streamline to be saved in the output \n'
                   'tractogram. If more than one --in_dpp_name was used, \n'
                   'enter the same number of --out_name values.')
    p.add_argument('out_tractogram', metavar='OUTPUT_FILE',
                   help='The file where the remaining streamlines \n'
                        'are saved.')
    p.add_argument('--endpoints_only', action='store_true', default=False,
                   help='If set, will only perform operation on endpoints \n'
                   'If not set, will perform operation on all streamline \n'
                   'points.')
    p.add_argument('--keep_all_dpp_dps', action='store_true',
                   help='If set, previous data_per_point will be preserved \n'
                   'in the output tractogram. Else, only --out_dpp_name \n'
                   'keys will be saved.')
    p.add_argument('--overwrite_dpp_dps', action='store_true',
                   help='If set, if --keep_all_dpp_dps is set and some \n'
                   '--out_dpp_name keys already existed in your \n'
                   ' data_per_point or data_per_streamline, allow \n'
                   ' overwriting old data_per_point.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    # Load the input files.
    logging.info("Loading file {}".format(args.in_tractogram))
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if len(sft.streamlines) == 0:
        logging.info("Input tractogram contains no streamlines. Exiting.")
        return

    if len(args.in_dpp_name) != len(args.out_name):
        parser.error('The number of in_dpp_names and out_names must be '
                     'the same.')

    # Check to see if there are duplicates in the out_names
    if len(args.out_name) != len(set(args.out_name)):
        parser.error('The output names (out_names) must be unique.')

    # Input name checks
    for in_dpp_name in args.in_dpp_name:
        # Check to see if the data per point exists.
        if in_dpp_name not in sft.data_per_point:
            logging.info('Data per point {} not found in input tractogram.'
                         .format(in_dpp_name))
            return

        # Check if first data_per_point is multivalued
        data_shape = sft.data_per_point[in_dpp_name][0].shape
        if args.operation == 'correlation' and len(data_shape) == 1:
            logging.info('Correlation operation requires multivalued data per '
                         'point. Exiting.')
            return

        if args.operation == 'correlation' and args.mode == 'dpp':
            logging.info('Correlation operation requires dps mode. Exiting.')
            return

        if not args.overwrite_dpp_dps:
            if in_dpp_name in args.out_name:
                logging.info('out_name {} already exists in input tractogram. '
                             'Set overwrite_dpp_dps or choose a different '
                             'out_name. Exiting.'.format(in_dpp_name))
                return

    data_per_point = {}
    data_per_streamline = {}
    for in_dpp_name, out_name in zip(args.in_dpp_name,
                                     args.out_name):
        # Perform the requested operation.
        if args.operation == 'correlation':
            logging.info('Performing {} across endpoint data and saving as '
                         'new dpp {}'.format(args.operation, out_name))
            new_dps = perform_pairwise_streamline_operation_on_endpoints(
                args.operation, sft, in_dpp_name)

            data_per_streamline[out_name] = new_dps
        elif args.mode == 'dpp':
            # Results in new data per point
            logging.info(
                'Performing {} on data from each streamine point '
                'and saving as new dpp {}'.format(
                    args.operation, out_name))
            new_dpp = perform_streamline_operation_per_point(
                args.operation, sft, in_dpp_name, args.endpoints_only)
            data_per_point[out_name] = new_dpp
        elif args.mode == 'dps':
            # Results in new data per streamline
            logging.info(
                'Performing {} across each streamline and saving resulting '
                'data per streamline {}'.format(args.operation, out_name))
            new_data_per_streamline = perform_operation_per_streamline(
                args.operation, sft, in_dpp_name, args.endpoints_only)
            data_per_streamline[out_name] = new_data_per_streamline

    if args.keep_all_dpp_dps:
        sft.data_per_streamline.update(data_per_streamline)
        sft.data_per_point.update(data_per_point)

        new_sft = sft
    else:
        new_sft = sft.from_sft(sft.streamlines, sft,
                               data_per_point=data_per_point,
                               data_per_streamline=data_per_streamline)

    # Print DPP names
    logging.info('New data per point names:')
    for key in new_sft.data_per_point.keys():
        logging.info(key)

    # Print DPS names
    logging.info('New data per streamline names:')
    for key in new_sft.data_per_streamline.keys():
        logging.info(key)

    # Save the new streamlines (and metadata)
    logging.info('Saving {} streamlines to {}.'.format(len(new_sft),
                                                       args.out_tractogram))
    save_tractogram(new_sft, args.out_tractogram,
                    bbox_valid_check=args.bbox_check)


if __name__ == "__main__":
    main()
