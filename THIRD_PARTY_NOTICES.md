# Source data and reuse information

No project-wide license for the author's code, manuscripts or generated results
has been selected in this release. This package does not assign a new license
to those materials. Public availability and an open-source license are separate
choices. See [GitHub's licensing documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

The input datasets retain their own terms. Their UCI (University of California,
Irvine) source pages identify the Creative Commons Attribution 4.0 International
license, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source information below was checked on 26 October 2026.

| Input | Attribution and permanent source | Included representation |
|---|---|---|
| Ionosphere | Sigillito, V., Wing, S., Hutton, L., and Baker, K. (1989). [UCI dataset, DOI 10.24432/C5W01B](https://doi.org/10.24432/C5W01B) | Study CSV, 351 samples and 34 features |
| Sonar | Sejnowski, T., and Gorman, R. (1988). [UCI dataset, DOI 10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q) | Study CSV, 208 samples and 60 features |
| Digits | Alpaydin, E., and Kaynak, C. (1998). [UCI dataset, DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49) | The 1797-sample scikit-learn subset of 8x8 images, serialized as NPZ |
| Wine | Aeberhard, S., and Forina, M. (1992). [UCI dataset, DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) | scikit-learn numerical input, 178 samples and 13 features, serialized as NPZ |
| Breast Cancer Wisconsin (Diagnostic) | Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). [UCI dataset, DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | scikit-learn numerical input, 569 samples and 30 features, serialized as NPZ |

The original study converted Ionosphere and Sonar to CSV and serialized the
scikit-learn inputs as NPZ (NumPy compressed array archives). These copies
preserve that study representation, including row order and target encoding.
Digits is a subset of the full UCI dataset, not the entire 5620-sample collection.
The original scikit-learn descriptions are retained under `datasets/sklearn`.
`datasets/uci/manifest.json` records the direct downloads and checksums for
Ionosphere and Sonar. No numerical input was changed for this release.

Python dependencies are installed separately from `requirements.txt` and retain
their respective licenses. The package does not vendor their implementations.

The bibliography lists the cited literature. Downloaded third-party article
full texts are not included in either distribution; use the source links in
[the references](docs/REFERENCES.md).
