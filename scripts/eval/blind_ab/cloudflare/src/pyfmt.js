// Python-compatible number formatting so exports match the Python server byte for byte.

const ROUND_DIGITS = 100; // Number.prototype.toFixed maximum; exact for the magnitudes we format.

function roundDecimalString(digits, ndigits) {
  // `digits` is the exact decimal expansion of a non-negative number ("123.4567...").
  const [intPart, fracPart = ""] = digits.split(".");
  const kept = fracPart.slice(0, ndigits).padEnd(ndigits, "0");
  const rest = fracPart.slice(ndigits);
  const asInt = BigInt(intPart + kept);
  let roundUp = false;
  if (rest.length && rest[0] > "5") roundUp = true;
  else if (rest.length && rest[0] === "5") {
    const exactHalf = /^50*$/.test(rest);
    roundUp = !exactHalf || asInt % 2n === 1n; // round half to even, like Python's round()
  }
  const value = (roundUp ? asInt + 1n : asInt).toString().padStart(ndigits + 1, "0");
  const cut = value.length - ndigits;
  return Number(`${value.slice(0, cut)}.${value.slice(cut)}`); // correctly rounded parse
}

/** Python's round(x, ndigits) for floats (correctly rounded, ties to even). */
export function pyRound(x, ndigits) {
  if (!Number.isFinite(x) || Math.abs(x) >= 1e21) return x; // toFixed switches to exponent form at 1e21
  const sign = x < 0 ? -1 : 1;
  const rounded = roundDecimalString(Math.abs(x).toFixed(ROUND_DIGITS), ndigits);
  return sign * rounded;
}

/** Python's str()/repr() of a float: shortest round-trip digits, Python's exponent rules. */
export function pyFloatStr(x) {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [mant, expStr] = x.toExponential().split("e");
  const exp = Number(expStr);
  const negative = mant.startsWith("-");
  const digits = mant.replace("-", "").replace(".", "");
  const sign = negative ? "-" : "";
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      const intPart = digits.slice(0, exp + 1).padEnd(exp + 1, "0");
      const frac = digits.slice(exp + 1) || "0";
      return `${sign}${intPart}.${frac}`;
    }
    return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
  }
  const lead = digits[0];
  const tail = digits.slice(1);
  const expAbs = String(Math.abs(exp)).padStart(2, "0");
  return `${sign}${lead}${tail ? `.${tail}` : ""}e${exp < 0 ? "-" : "+"}${expAbs}`;
}

/** Format a value the way Python's csv module would write it. */
export function pyCsvValue(value, isFloat) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return isFloat ? pyFloatStr(value) : String(value);
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

/** json.dumps(obj, ensure_ascii=False, sort_keys=True) for a flat record of strings and floats. */
export function pyJsonFlat(record, floatKeys) {
  const parts = Object.keys(record).sort().map((key) => {
    const value = record[key];
    const encoded = floatKeys.has(key) && typeof value === "number" ? pyFloatStr(value) : JSON.stringify(value);
    return `${JSON.stringify(key)}: ${encoded}`;
  });
  return `{${parts.join(", ")}}`;
}
