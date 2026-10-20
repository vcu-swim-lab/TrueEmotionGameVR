using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace VcuSeCenter.Calibration
{
    /// <summary>Thrown when a document is not the JSON this reader accepts.</summary>
    public sealed class JsonParseException : Exception
    {
        public JsonParseException(string message) : base(message) { }
    }

    /// <summary>
    /// A small strict JSON reader, used only to load the population artifact.
    /// </summary>
    /// <remarks>
    /// Unity's `JsonUtility` cannot represent the artifact -- it has no dictionary support
    /// and silently returns default values for anything it fails to map, which is precisely
    /// the failure mode a provenance check exists to prevent. `System.Text.Json` is not
    /// available on the .NET Standard 2.1 profile Unity builds against. So the artifact
    /// reader owns a parser rather than a dependency.
    ///
    /// It is deliberately strict: trailing data, unterminated containers, unquoted keys and
    /// malformed numbers all raise. A validator that quietly accepts a broken file is worse
    /// than no validator, because it converts an obvious failure into a plausible-looking
    /// number.
    ///
    /// Numbers are parsed as `double` with round-trip semantics, so a value written by
    /// Python's shortest round-trip `repr` reads back as the identical double. That is what
    /// lets the conversion hash -- computed over IEEE-754 bit patterns -- agree across the
    /// two languages.
    /// </remarks>
    public static class MiniJson
    {
        public static object Parse(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));

            int index = 0;

            // A UTF-8 BOM survives some editors. Skipping it here keeps the failure at
            // "this artifact is wrong" rather than "character 0 is unexpected".
            if (text.Length > 0 && text[0] == '﻿') index = 1;

            SkipWhitespace(text, ref index);
            object value = ParseValue(text, ref index);
            SkipWhitespace(text, ref index);

            if (index != text.Length)
            {
                throw new JsonParseException(
                    $"trailing content after the top-level value at character {index}");
            }

            return value;
        }

        // ------------------------------------------------------------------ typed access

        public static Dictionary<string, object> AsObject(object value, string what)
        {
            if (value is Dictionary<string, object> map) return map;
            throw new JsonParseException($"{what}: expected an object, found {Describe(value)}");
        }

        public static List<object> AsArray(object value, string what)
        {
            if (value is List<object> list) return list;
            throw new JsonParseException($"{what}: expected an array, found {Describe(value)}");
        }

        public static string AsString(object value, string what)
        {
            if (value is string text) return text;
            throw new JsonParseException($"{what}: expected a string, found {Describe(value)}");
        }

        public static double AsDouble(object value, string what)
        {
            if (value is double number) return number;
            throw new JsonParseException($"{what}: expected a number, found {Describe(value)}");
        }

        public static object Field(Dictionary<string, object> map, string key, string what)
        {
            if (map.TryGetValue(key, out object value)) return value;
            throw new JsonParseException($"{what}: no '{key}' field");
        }

        public static string Describe(object value)
        {
            if (value == null) return "null";
            if (value is Dictionary<string, object>) return "an object";
            if (value is List<object>) return "an array";
            if (value is string) return "a string";
            if (value is double) return "a number";
            if (value is bool) return "a boolean";
            return value.GetType().Name;
        }

        // ---------------------------------------------------------------------- parsing

        private static object ParseValue(string text, ref int index)
        {
            if (index >= text.Length)
            {
                throw new JsonParseException("the document ended where a value was expected");
            }

            char c = text[index];
            switch (c)
            {
                case '{': return ParseObject(text, ref index);
                case '[': return ParseArray(text, ref index);
                case '"': return ParseString(text, ref index);
                case 't': Expect(text, ref index, "true"); return true;
                case 'f': Expect(text, ref index, "false"); return false;
                case 'n': Expect(text, ref index, "null"); return null;
                default: return ParseNumber(text, ref index);
            }
        }

        private static Dictionary<string, object> ParseObject(string text, ref int index)
        {
            var map = new Dictionary<string, object>(StringComparer.Ordinal);
            index++; // '{'
            SkipWhitespace(text, ref index);

            if (Peek(text, index) == '}') { index++; return map; }

            while (true)
            {
                SkipWhitespace(text, ref index);
                if (Peek(text, index) != '"')
                {
                    throw new JsonParseException($"expected a quoted key at character {index}");
                }

                string key = ParseString(text, ref index);
                SkipWhitespace(text, ref index);

                if (Peek(text, index) != ':')
                {
                    throw new JsonParseException($"expected ':' after key '{key}' at character {index}");
                }
                index++;

                SkipWhitespace(text, ref index);
                map[key] = ParseValue(text, ref index);
                SkipWhitespace(text, ref index);

                char next = Peek(text, index);
                if (next == ',') { index++; continue; }
                if (next == '}') { index++; return map; }

                throw new JsonParseException($"expected ',' or '}}' at character {index}");
            }
        }

        private static List<object> ParseArray(string text, ref int index)
        {
            var list = new List<object>();
            index++; // '['
            SkipWhitespace(text, ref index);

            if (Peek(text, index) == ']') { index++; return list; }

            while (true)
            {
                SkipWhitespace(text, ref index);
                list.Add(ParseValue(text, ref index));
                SkipWhitespace(text, ref index);

                char next = Peek(text, index);
                if (next == ',') { index++; continue; }
                if (next == ']') { index++; return list; }

                throw new JsonParseException($"expected ',' or ']' at character {index}");
            }
        }

        private static string ParseString(string text, ref int index)
        {
            index++; // opening quote
            var sb = new StringBuilder();

            while (true)
            {
                if (index >= text.Length)
                {
                    throw new JsonParseException("the document ended inside a string");
                }

                char c = text[index++];
                if (c == '"') return sb.ToString();

                if (c != '\\') { sb.Append(c); continue; }

                if (index >= text.Length)
                {
                    throw new JsonParseException("the document ended inside an escape sequence");
                }

                char escape = text[index++];
                switch (escape)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (index + 4 > text.Length)
                        {
                            throw new JsonParseException("truncated \\u escape");
                        }
                        sb.Append((char)ushort.Parse(text.Substring(index, 4),
                                                     NumberStyles.HexNumber,
                                                     CultureInfo.InvariantCulture));
                        index += 4;
                        break;
                    default:
                        throw new JsonParseException($"unknown escape '\\{escape}' at character {index - 1}");
                }
            }
        }

        private static double ParseNumber(string text, ref int index)
        {
            int start = index;

            if (Peek(text, index) == '-') index++;
            while (index < text.Length && char.IsDigit(text[index])) index++;

            if (Peek(text, index) == '.')
            {
                index++;
                while (index < text.Length && char.IsDigit(text[index])) index++;
            }

            char exponent = Peek(text, index);
            if (exponent == 'e' || exponent == 'E')
            {
                index++;
                char sign = Peek(text, index);
                if (sign == '+' || sign == '-') index++;
                while (index < text.Length && char.IsDigit(text[index])) index++;
            }

            string literal = text.Substring(start, index - start);

            if (literal.Length == 0 || literal == "-")
            {
                throw new JsonParseException($"expected a number at character {start}");
            }

            // `Float` with the invariant culture: a Turkish locale must not turn "0.4" into 4,
            // which is the same class of bug the CSV writer guards against.
            //
            // The two failures below are one failure wearing two faces. Given a magnitude too
            // large for a double, .NET Core returns positive or negative infinity while Mono
            // -- which is what Unity runs -- rejects the parse outright. Both mean the file
            // holds a value this reader cannot represent, so both are reported identically
            // rather than leaving the artifact's behaviour to depend on which runtime opened
            // it. Found by the Unity EditMode run disagreeing with `dotnet test` on exactly
            // this input.
            bool parsed = double.TryParse(literal, NumberStyles.Float,
                                          CultureInfo.InvariantCulture, out double value);

            if (!parsed || double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new JsonParseException(
                    $"'{literal}' at character {start} is not a finite number");
            }

            return value;
        }

        private static void Expect(string text, ref int index, string literal)
        {
            if (index + literal.Length > text.Length
                || string.CompareOrdinal(text, index, literal, 0, literal.Length) != 0)
            {
                throw new JsonParseException($"expected '{literal}' at character {index}");
            }

            index += literal.Length;
        }

        private static char Peek(string text, int index)
            => index < text.Length ? text[index] : '\0';

        private static void SkipWhitespace(string text, ref int index)
        {
            while (index < text.Length)
            {
                char c = text[index];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') index++;
                else break;
            }
        }
    }
}
