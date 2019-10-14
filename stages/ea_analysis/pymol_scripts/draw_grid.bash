SED_EXPR="s/FOLDER/${1//\//\\\/}/"
PYMOL_SCRIPT=draw_grid_"$$".pml
sed "$SED_EXPR" draw_grid.pml > "$PYMOL_SCRIPT"
pymol "$PYMOL_SCRIPT"
rm "$PYMOL_SCRIPT"
