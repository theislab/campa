import re

from enchant.tokenize import Filter


class ModnameFilter(Filter):
    """Ignore module names."""

    _pat = re.compile(r"campa\.(tl|pl|data|utils|constants).+")

    def _skip(self, word: str) -> bool:
        return self._pat.match(word) is not None


class SignatureFilter(Filter):
    """Ignore function signature artifacts."""

    # underscore separated words
    _pat = re.compile(r"[a-z]+(_[a-z]+)+")
    # _pat = re.compile(r"(\(.*\[.*,.*\]\)|\([a-z_]+(, [a-z_]+)*\))")

    def _skip(self, word: str) -> bool:
        if self._pat.match(word) is not None:
            print(f"skipping {word}")
            return True
        else:
            return word in (
                "95th",
                "groupby",
                "concat",
                "imgs",
                "colors[",
                "num,",
                "[frac",
                "img[",
                "df[",
                "exps[",
                "groupby[",
                "desc[",
                "(objs)",
                "fname",
                "num",
                "frac",
                "objs",
                "params",
                "agg",
            )


class MDLinkFilter(Filter):
    """Ignore markdown links enclosed in <...>"""

    _pat = re.compile(r"<.+>")

    def _skip(self, word: str) -> bool:
        return self._pat.match(word) is not None
