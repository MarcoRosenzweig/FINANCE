from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument('--foo', action='store_true', help='foo help')

subparsers = parser.add_subparsers(help="sub-command help.")

# create the parser for the "a" command
parser_a = subparsers.add_parser('RenameVariables', help='a help')
parser_a.add_argument('variableName', type=str, help='variablesName', nargs="+")
parser_a.add_argument("-nvn", "--newVariableName", type=str, nargs="+")

# create the parser for the "b" command
parser_b = subparsers.add_parser('b', help='b help')
parser_b.add_argument('--baz', choices='XYZ', help='baz help')

command_line_args = ["--foo", "RenameVariables", "e", "f", "-nvn", "1", "3"]
args = parser.parse_args(command_line_args)
print(args.variableName)
args = parser.parse_args(sys.argv[1:])
print(args)