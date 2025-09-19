#!/usr/bin/bash
gmt begin moho_plot_Gravity png
    # Set region and projection
    gmt set MAP_FRAME_TYPE=plain
    gmt basemap -R65/112/20/47 -JM15c -BWeSn -Bxa5f1 -Bya2f1
    
    # Convert XYZ data to grid file
    gmt xyz2grd GravityMoho.txt -Gmoho.nc -I0.25/0.25 -R65/112/20/47
    
    # Plot grid with color palette
    gmt makecpt -Cseis -T-10/80/5 -Z -I  # Reverse seismic colormap for depth
    gmt grdimage moho.nc -C -t50       # Adjust transparency if needed
    
    # Add colorbar and labels
    gmt colorbar -DJBC+w10c+o0/1c -Bxaf+l"Moho Depth (km)" -I
    
    # Overlay coastlines/borders
    gmt coast -W1p,black -Df -N1/1p,black
    
    # Add title
    echo "Moho_Gravity" | gmt text -F+f16p,Helvetica-Bold,black
gmt end show