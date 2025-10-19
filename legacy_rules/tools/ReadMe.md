Как пользоваться

Автопоиск:
python rules_linter.py

С явными путями:
python rules_linter.py --paths rules/common.yaml themes/biomed/rules.yaml themes/biomed/triggers.yaml

JSON-отчёт (для CI):
python rules_linter.py --json lint_report.json

«Жёсткий режим» (warnings = ошибки):
python rules_linter.py --strict