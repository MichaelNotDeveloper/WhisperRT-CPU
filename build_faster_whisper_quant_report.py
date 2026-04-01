from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent
OUT_NOTEBOOK = ROOT / "faster_whisper_quant_report_executed.ipynb"


def build_notebook():
    return new_notebook(
        cells=[
            new_markdown_cell(
                "# Faster-Whisper Quantization Experiments\n\n"
                "Этот notebook содержит только эксперименты по квантизации для `faster-whisper`.\n\n"
                "Успешно отработали quant-only варианты:\n"
                "- `faster-whisper|CPU|int8`\n"
                "- `faster-whisper|CPU|int8_float32`\n"
                "\n"
                "Дополнительно были попробованы GPU-варианты:\n"
                "- `faster-whisper|GPU|int8_float16`\n"
                "- `faster-whisper|GPU|int8`\n"
                "\n"
                "Обе GPU-конфигурации на этом сервере завершились `CUDA out of memory`, поэтому итоговая таблица `RTF` ниже содержит только успешные quant-only запуски.\n\n"
                "Все ячейки выполнены, а тяжелый прогон был сделан заранее на сервере."
            ),
            new_markdown_cell("## Результаты"),
            new_code_cell(
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "\n"
                "ROOT = Path.cwd()\n"
                "results = pd.read_csv(ROOT / 'faster_whisper_quant_results.csv')\n"
                "results.sort_values('rtf')"
            ),
            new_markdown_cell("## RTF по экспериментам"),
            new_code_cell(
                "results[['name', 'rtf', 'wer', 'wall_time_sec', 'status']].sort_values('rtf')"
            ),
            new_markdown_cell("## Неуспешные GPU-попытки"),
            new_code_cell(
                "import pandas as pd\n"
                "gpu_attempts = pd.DataFrame([\n"
                "    {'name': 'faster-whisper|GPU|int8_float16', 'status': 'error', 'error': 'CUDA failed with error out of memory'},\n"
                "    {'name': 'faster-whisper|GPU|int8', 'status': 'error', 'error': 'CUDA failed with error out of memory'},\n"
                "])\n"
                "gpu_attempts"
            ),
            new_markdown_cell(
                "## Вывод\n\n"
                "Этот notebook специально ограничен только quantization-ветками `faster-whisper`, без pruning, compile, ONNX и других не-quant экспериментов."
            ),
        ]
    )


def main():
    nb = build_notebook()
    client = NotebookClient(nb, timeout=120, kernel_name='python3')
    executed = client.execute()
    nbformat.write(executed, OUT_NOTEBOOK)
    print(OUT_NOTEBOOK)


if __name__ == '__main__':
    main()
