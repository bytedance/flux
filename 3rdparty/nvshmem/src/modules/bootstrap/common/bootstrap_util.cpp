/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                                        // for assert
#include <stdio.h>                                                         // for printf, fprintf
#include <stdlib.h>                                                        // for NULL, size_t
#include <string.h>                                                        // for memcpy, memset
#include <string>                                                          // for string, basic...
#include "bootstrap_util.h"                                                // for bootstrap_uti...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for BOOTSTRAP_OPT...

/* Wrap 'str' to fit within 'wraplen' columns. Will not break a line of text
 * with no whitespace that exceeds the allowed length. After each line break,
 * insert 'indent' string (if provided).  Caller must free the returned buffer.
 */
char *bootstrap_util_wrap_string(const char *str, const size_t wraplen, const char *indent,
                                 const int strip_backticks) {
    const size_t indent_len = indent != NULL ? strlen(indent) : 0;
    size_t str_len = 0, line_len = 0, line_breaks = 0;
    char *str_s = NULL;

    /* Count characters and newlines */
    for (const char *s = str; *s != '\0'; s++, str_len++)
        if (*s == '\n') ++line_breaks;

    /* Worst case is wrapping at 1/2 wraplen plus explicit line breaks. Each
     * wrap adds an indent string. The newline is either already in the source
     * string or replaces a whitespace in the source string */
    const size_t out_len = str_len + 1 + (2 * (str_len / wraplen + 1) + line_breaks) * indent_len;
    char *out = (char *)calloc(out_len, sizeof(char));
    char *str_p = (char *)str;
    std::string statement = "";
    std::string last_word = "";

    if (out == NULL) {
        fprintf(stderr, "%s:%d Unable to allocate output buffer\n", __FILE__, __LINE__);
        return NULL;
    }

    while (*str_p != '\0' &&
           /* avoid overflowing out */ statement.length() < (out_len - 1)) {
        /* Remember location of last space */
        if (*str_p == ' ') {
            str_s = str_p;
        }
        /* Wrap here if there is a newline */
        else if (*str_p == '\n') {
            str_s = str_p;
            statement += "\n"; /* Append newline and indent */
            if (indent) {
                statement += indent;
            }
            str_p++;
            str_s = NULL;
            line_len = 0;
            continue;
        }

        /* Remove backticks from the input string */
        else if (*str_p == '`' && strip_backticks) {
            str_p++;
            continue;
        }

        /* Reached end of line, try to wrap */
        if (line_len >= wraplen) {
            if (str_s != NULL) {
                str_p = str_s; /* Jump back to last space */
                size_t found =
                    statement.find_last_of(" "); /* Find the last token, remove it from statement as
                                                    it will be appended subsequently */
                last_word = statement.substr(found + 1);
                statement.erase(found, found + 1 + last_word.length());
                statement += "\n"; /* Append newline and indent */
                if (indent) {
                    statement += indent;
                }
                str_p++;
                str_s = NULL;
                line_len = 0;
                continue;
            }
        }
        statement += (*str_p);
        str_p++;
        line_len++;
    }

    memset(out, '\0', out_len);
    memcpy(out, statement.c_str(), statement.length());
    return out;
}

void bootstrap_util_print_header(int style, const char *h) {
    switch (style) {
        case BOOTSTRAP_OPTIONS_STYLE_INFO:
            printf("%s:\n", h);
            break;
        case BOOTSTRAP_OPTIONS_STYLE_RST:
            printf("%s\n", h);
            for (const char *c = h; *c != '\0'; c++) putchar('~');
            printf("\n\n");
            break;
        default:
            assert(0);
    }
}
