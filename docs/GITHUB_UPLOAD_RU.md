# Загрузка проекта на GitHub через браузер

Комплект подготовлен 26 октября 2026 года. Создание репозитория и загрузка
выполняются владельцем учетной записи. В комплекте нет вымышленного адреса
репозитория, DOI или утверждения о состоявшейся публикации.

1. Распакуйте `ocba_github_upload_20261026.zip`. Откройте папку `repository`.
2. Создайте на GitHub пустой репозиторий. Например, `pso-ocba-noise-budget`.
   Выберите нужную видимость. Общая лицензия коду и статье пока не назначена;
   условия исходных датасетов приведены в `THIRD_PARTY_NOTICES.md`.
3. Первый заход: откройте Add file / Upload files и перетащите содержимое
   `repository`, кроме папки `results`. Сохраняйте структуру вложенных папок.
   Загружать нужно файлы и папки изнутри, а не саму папку `repository` и не ZIP.
   Включите `.gitignore` и `.gitattributes`. Завершите Commit changes или
   Propose changes, как предложит интерфейс.
4. Второй заход: снова Add file / Upload files. Перетащите целиком папку
   `results` из `repository`, чтобы в репозитории получился путь `results/...`.
   Завершите сохранение. Дождитесь окончания обоих заходов перед проверкой.
5. Проверьте, что README открывается на главной странице, есть четыре Word-файла,
   каталог `datasets`, Python-код и `results`. Внешние `upload_plan.json` и
   эта инструкция из корня ZIP помогают загрузке; их отдельно загружать не нужно.
6. Для контрольной проверки скачайте весь репозиторий через Code / Download ZIP,
   распакуйте и откройте PowerShell в каталоге с `experiment.py`:

```powershell
py -3.13 prepare_data.py
py -3.13 verify_reviewer_package.py
```

Установка зависимостей и пересчет статистики:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_reviewer_package.py --recompute
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Этот пересчет использует сохраненные результаты и три примера трасс. Он не
запускает заново всю серию из 351216000 оценок. Полные первичные трассы находятся
в отдельном сохраненном архиве; в GitHub-комплект они не входят.

GitHub допускает до 25 MiB на файл и до 100 файлов за одну загрузку через
браузер. В `upload_plan.json` записаны точные списки и количество файлов двух
заходов. Каждый файл подготовленного репозитория укладывается в этот предел.
Одна большая CSV-таблица хранится как `.csv.gz`. `prepare_data.py` проверяет
загруженные файлы и восстанавливает ее с точным совпадением SHA-256 и размера.
Интернет для восстановления не требуется. Измененный существующий CSV
скрипт не заменяет. Научные модули читают прежний путь CSV после восстановления.

Загруженный ZIP GitHub автоматически не распаковывает. Не публикуйте вместо
дерева проекта один ZIP. Полные чужие статьи, старые редакции, кеши, окружения
и журналы разработки в этот комплект не включены.

Проверенные официальные источники:

- [Загрузка файлов и ограничения браузера](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).
- [Крупные файлы](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- [Лицензирование репозитория](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
