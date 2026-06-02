import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--save-lr-dir', type=str, default=None,
                    help='Directory for generated low-resolution inputs.')
parser.add_argument('--report-file', type=str, default=None,
                    help='Path to JSON report. Defaults to save-dir/eval_report.json or ./eval_report.json.')
parser.add_argument('--eval-set', type=str, default='DIV2K_val')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--eval-scale', type=int, default=4)
parser.add_argument('--patch-size-dec', type=int, default=256,
                    help='Decoder patch size. Lower values reduce memory usage.')
parser.add_argument('--no-geo-ensemble', action='store_true')
parser.add_argument('--y-only', action='store_true', help='Only evaluate Y channel of YCbCr image')
