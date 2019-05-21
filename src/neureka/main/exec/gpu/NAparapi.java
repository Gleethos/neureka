
package neureka.main.exec.gpu;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;

import javax.swing.JComponent;
import javax.swing.JFrame;

//package com.amd.aparapi.sample.e_add;
//import com.amd.aparapi.Kernel;
//import com.amd.aparapi.Range;
import com.aparapi.Kernel;
import com.aparapi.ProfileInfo;
import com.aparapi.Range;
//import com.aparapi.internal.kernel.*;

public class NAparapi{
   /**
    * An Aparapi Kernel implementation for creating a scaled view of the mandelbrot e_set.
    */
	
	   /** User selected zoom-in point on the Mandelbrot view. */
	   public static volatile Point to = null;

	   //@SuppressWarnings("serial")
	   public void start() {

	      final JFrame frame = new JFrame("Calc Frame");
	      /** Width of Mandelbrot view. */
	      final int width = 1200;
	      /** Height of Mandelbrot view. */
	      final int height = 968;
	      /** Mandelbrot image height. */
	      final Range range = Range.create(width * height);

	      /** Image for Mandelbrot view. */
	      final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	      final BufferedImage offscreen = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	      // Draw Mandelbrot image
	      final JComponent viewer = new JComponent(){
	         @Override public void paintComponent(Graphics g) {

	            g.drawImage(image, 0, 0, width, height, this);
	         }
	      };

	      // Set the size of JComponent which displays Mandelbrot image
	      viewer.setPreferredSize(new Dimension(width, height));

	      final Object doorBell = new Object();

	      // Mouse listener which reads the user clicked zoom-in point on the Mandelbrot view 
	      viewer.addMouseListener(new MouseAdapter(){
	         @Override public void mouseClicked(MouseEvent e) {
	            to = e.getPoint();
	            synchronized (doorBell) {
	               doorBell.notify();
	            }
	         }
	      });

	      // Swing housework to create the frame
	      frame.getContentPane().add(viewer);
	      frame.pack();
	      frame.setLocationRelativeTo(null);
	      frame.setVisible(true);

	      // Extract the underlying RGB buffer from the image.
	      // Pass this to the kernel so it operates directly on the RGB buffer of the image
	      final int[] rgb = ((DataBufferInt) offscreen.getRaster().getDataBuffer()).getData();
	      final int[] imageRgb = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
	      // Create a Kernel passing the size, RGB buffer and the palette.
	      final MandelKernel kernel = new MandelKernel(width, height, rgb);

	      final float defaultScale = 3f;

	      // Set the default scale and offset, execute the kernel and force a repaint of the viewer.
	      kernel.setScaleAndOffset(defaultScale, -1f, 0f);
	      kernel.execute(range);

	      System.arraycopy(rgb, 0, imageRgb, 0, rgb.length);
	      viewer.repaint();

	      // Report target execution mode: GPU or JTP (Java Thread Pool).
	      System.out.println("Execution mode=" + kernel.getExecutionMode());

	      // Window listener to dispose Kernel resources on user exit.
	      frame.addWindowListener(new WindowAdapter(){
	         @Override public void windowClosing(WindowEvent _windowEvent) {
	            kernel.dispose();
	            System.exit(0);
	         }
	      });
	      
	      
	      float scale = defaultScale;
	         float tox;
	         float toy;
	         float x = -1f;
	         float y = 0f;
	         int counter= 0;
	      
	      // Wait until the user selects a zoom-in point on the Mandelbrot view.
	      while (true) {
	    	  counter++;
	         // Wait for the user to click somewhere
	         while (to == null) {
	            synchronized (doorBell) {
	               try {
	                  doorBell.wait();
	               } catch (final InterruptedException ie) {
	                  ie.getStackTrace();
	               }
	            }
	         }   
	         
	         
	         tox = ((float) (to.x - (width / 2)) / width) * scale;
	         toy = ((float) (to.y - (height / 2)) / height) * scale;

	         // This is how many frames we will display as we zoom in and out.
	         final int frames = 128;
	         final long startMillis = System.currentTimeMillis();
	         for (int sign = -1; sign < 0; sign ++) {
	            for (int i = 0; i < (frames - 4); i++) {
	               scale = scale + ((sign * scale) / frames);
	               x = x - (sign * (tox / frames));
	               y = y - (sign * (toy / frames));

	               // Set the scale and offset, execute the kernel and force a repaint of the viewer.
	               kernel.setScaleAndOffset(scale, x, y);
	               kernel.setModifier((float)Math.sin((((double)counter)/100)));
	               kernel.execute(range);
	               final ArrayList<ProfileInfo> profileInfo = (ArrayList<ProfileInfo>) kernel.getProfileInfo();
	               if ((profileInfo != null) && (profileInfo.size() > 0)) {
	                  for (final ProfileInfo p : profileInfo) {
	                     System.out.print(" " + p.getType() + " " + p.getLabel() + " " + (p.getStart() / 1000) + " .. "
	                           + (p.getEnd() / 1000) + " " + ((p.getEnd() - p.getStart()) / 1000) + "us");
	                  }
	                  System.out.println();
	               }

	               System.arraycopy(rgb, 0, imageRgb, 0, rgb.length);
	               viewer.repaint();
	            }
	         }

	         final long elapsedMillis = System.currentTimeMillis() - startMillis;
	         System.out.println("FPS = " + ((frames * 1000) / elapsedMillis));

	         // Reset zoom-in point.
	         to = null;

	      }

	   }
	
	
	//KERNEL:
   //=======================================================================================================================
   public static class MandelKernel extends Kernel{
      /** RGB buffer used to copy the Mandelbrot image. This buffer holds (width * height) RGB values. */
      final private int rgb[];
      /** Mandelbrot image width. */
      final private int width;
      /** Mandelbrot image height. */
      final private int height;
      /** Maximum iterations for Mandelbrot. */
      final private int maxIterations = 600;//64
      /** Palette which maps iteration values to RGB values. */
      @Constant final private int pallette[] = new int[maxIterations + 1];
      /** Mutable values of scale, offsetx and offsety so that we can modify the zoom level and position of a view. */
      private float scale = .0f;
      private float offsetx = .0f;
      private float offsety = .0f;
      private float modifier = 0;
      /**
       * Initialize the Kernel.
       * @param _width Mandelbrot image width
       * @param _height Mandelbrot image height
       * @param _rgb Mandelbrot image RGB buffer
       * @param _pallette Mandelbrot image palette
       */
      public MandelKernel(int _width, int _height, int[] _rgb) {
         //Initialize palette values
         for (int i = 0; i < maxIterations; i++) {
            final float h = i / (float) maxIterations;
            final float b = 1.0f - (h * h);
            pallette[i] = Color.HSBtoRGB(h, 1f, b);
         }
         width = _width;
         height = _height;
         rgb = _rgb;
      }

      public int getCount(float x, float y) {
         int count = 0;
         float zx = x;
         float zy = y;
         float new_zx = 0f;
         // Iterate until the algorithm converges or until maxIterations are reached.
         while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
            new_zx = ((zx * zx) - (zy * zy))*(1f) + x;
            //zy = (float)Math.pow((((zy * zx))*2f + y),0.5);
            zy = ((2) * zx * zy) + y;
            //zx = (float)Math.pow(new_zx, 0.5f);
            zx = new_zx;
            count++;
         }
         return count;
      }

      @Override public void run() {

         /** Determine which RGB value we are going to process (0..RGB.length). */
         final int gid = getGlobalId();
         /** Translate the gid into an x an y value. */
         final float x = ((((gid % width) * scale) - ((scale / 2) * width)) / width) + offsetx;
         final float y = ((((gid / width) * scale) - ((scale / 2) * height)) / height) + offsety;
         int count = getCount(x, y);
         // Pull the value out of the palette for this iteration count.
         rgb[gid] = pallette[count];
      }
      public void setScaleAndOffset(float _scale, float _offsetx, float _offsety) {
         offsetx = _offsetx;
         offsety = _offsety;
         scale = _scale;
      }
      public void setModifier(float mod) {modifier = mod;}
   }
   //=======================================================================================================================
   //KERNEL END


}
