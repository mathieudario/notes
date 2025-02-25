#!/bin/bash
CONTENTDIR="content"
ASSETSDIR="assets"
BUILDDIR="build"
FILENAME="OOD and OMS paradigms in object detection"

pdf() {
    mkdir "${BUILDDIR}" -p
    echo "Creating pdf output file ..."
    pandoc "${CONTENTDIR}/${FILENAME}.md" \
        --resource-path="${CONTENTDIR}" \
        --csl="${ASSETSDIR}/citation-style.csl" \
        --from="markdown+tex_math_single_backslash+tex_math_dollars+raw_tex" \
        --to="latex" \
        --output="${BUILDDIR}/${FILENAME}.pdf" \
        --pdf-engine="pdflatex" \
        --bibliography="${CONTENTDIR}/bibliography.bib" 
}

# Allows to call a function based on arguments passed to the script
# Example: `./build.sh pdf_print`
$*
