# Проверка и обновление репозитория

Публичный проект: [pso-ocba-noise-budget](https://github.com/Inkizhekov/pso-ocba-noise-budget).
Код и техническая документация используют MIT. Статьи, результаты и входные
датасеты имеют отдельные условия в [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Проверка скачанного проекта

Скачайте весь репозиторий через Code / Download ZIP, распакуйте его и откройте
PowerShell в каталоге с `experiment.py`:

```powershell
py -3.13 prepare_data.py
py -3.13 verify_reviewer_package.py
```

Эти команды используют стандартную библиотеку Python. Проверяются неизменные
научные материалы: `data/`, `datasets/`, `results/`, файлы DOCX и PDF. Большая
таблица из `.csv.gz` восстанавливается с проверкой размера и SHA-256
(Secure Hash Algorithm, 256-bit). Измененный существующий CSV не заменяется.
Правка README или кода не блокирует подготовку сохраненных данных.

Для проверки точных байтов выпуска, включая код и документацию:

```powershell
py -3.13 prepare_data.py --verify-only --strict-release
py -3.13 verify_reviewer_package.py --strict-release
```

Вторая команда требует предварительного восстановления CSV. Два manifest
исключают себя и друг друга из контрольных сумм во избежание циклической
зависимости; их собственные байты фиксирует Git-коммит. Внешний файл
`.zip.sha256` для автоматически сформированного GitHub ZIP не предоставляется.

После установки научных зависимостей:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_reviewer_package.py --recompute
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Пересчет использует готовые результаты и три примера трасс. Все 351216000
наблюдений заново не выполняются. [Полный архив пока не опубликован](DATA_AVAILABILITY.md).

## Обновление кода и документации

В Git-копии после проверки изменений выполните:

```powershell
py -3.13 refresh_release_manifest.py --check
py -3.13 refresh_release_manifest.py --write
py -3.13 prepare_data.py --verify-only --strict-release
py -3.13 verify_reviewer_package.py --strict-release
git diff --stat
```

`--check` сообщает об устаревшем перечне или суммах, ничего не записывая.
`--write` сначала сверяет прежние научные материалы и отказывается обновлять
manifest, если любой из них изменен или отсутствует. Затем обновляются суммы
кода и документации. Скрипт учитывает отслеживаемые Git файлы и новые файлы,
не исключенные `.gitignore`; проверьте их список перед записью. Он работает
и до восстановления большого CSV, проверяя логическое содержимое gzip.

Добавление нового научного материала требует явного пути, например:

```powershell
py -3.13 refresh_release_manifest.py --write --add-research manuscripts/new_article.pdf
```

Это разрешает зафиксировать только новый файл. Изменение уже зафиксированного
материала не разрешается даже с `--add-research`. Прежние архивы и результаты
не переписывайте. Новые поиски сохраняйте в `results/my_...`, `results/new_...`
или `reviewer_checks/`. Эти каталоги исключены из публикации.

Файл `.github/workflows/checks.yml` задает автоматическую проверку на Windows
с Python 3.13.3: подготовка данных, тесты, небольшой технический поиск и
пересчет сохраненной статистики. Строгая проверка байтов выпуска выполняется
отдельно, чтобы обычная правка документации не блокировала научные проверки.

## Загрузка изменений через браузер

Через Add file / Upload files загрузите измененные файлы с сохранением путей.
Включите оба обновленных manifest. При создании файла workflow используйте
полный путь `.github/workflows/checks.yml` в Add file / Create new file.
После Commit changes проверьте раздел Actions. Не загружайте окружения,
кеши, новые рабочие результаты или большой полный архив вместо дерева кода.

GitHub не распаковывает загруженный ZIP автоматически. Для ограничений
браузерной загрузки используйте [официальную инструкцию GitHub](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).
