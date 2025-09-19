#!/usr/bin/bash
gmt begin moho_plot_RF_dat png
    # Set region and projection
    gmt set MAP_FRAME_TYPE=plain
    gmt basemap -R74/110/26/42 -JM15c -BWeSn -Bxa5f1 -Bya2f1
    
    # Convert XYZ data to grid file
    awk '$3<100{print $1,$2,$3}' Moho_RF.dat | gmt xyz2grd -Gmoho.nc -I0.25/0.25 -R74/110/26/42
    
    # Plot grid with color palette
    gmt makecpt -Cseis -T30/75/5 -Z -I -D  # Reverse seismic colormap for depth
    gmt grdimage moho.nc -C -t50       # Adjust transparency if needed
    
    # Add colorbar and labels
    gmt colorbar -DJBC+w10c+o0/1c -Bxaf+l"Moho Depth (km)" -I
    
    # Overlay coastlines/borders
    gmt coast -W1p,black -Df -N1/1p,black
    
    # Add title
    echo "Moho_RF_dat" | gmt text -F+f16p,Helvetica-Bold,black
gmt end show