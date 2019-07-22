package neureka.ngui;

import java.awt.Color;
import java.util.List;
import java.util.Random;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;

public class N3DSpace {

	NPanel Surface;
	float[] distance = {0,0,0};
	NPointTracer PointTracer;
	
	public N3DSpace(int pointCount) 
	{
		Surface = new NPanel();
		
		Random dice = new Random();
		float[] points = new float[3*pointCount];
		for(int i=0; i<pointCount; i++) 
		{
			
			points[i*3+0] = 1000*dice.nextFloat()-500;
			points[i*3+1] = 1000*dice.nextFloat()-500;
			points[i*3+2] = 1000*dice.nextFloat()-500;
			//System.out.println("x:"+points[i*3+0]);
			//System.out.println("y:"+points[i*3+1]);
			//System.out.println("z:"+points[i*3+2]);
		}
		PointTracer = new NPointTracer(points);
		
		Surface.setPaintAction(
			(surface, brush)->
			{
				brush.setColor(Color.WHITE);
				brush.fillOval((int)surface.realX(surface.lastSenseX())-5, (int)surface.realY(surface.lastSenseY())-5, 10, 10);
				brush.setColor(Color.GREEN);
				brush.fillOval((int)(-1000), (int)(-1000), (2000), (2000));
				brush.setColor(Color.DARK_GRAY);
				brush.fillOval((int)(-500), (int)(-500), (1000), (1000));
				
				int x = 0;
				int y = 0;
				for(int Pi=0; Pi<PointTracer.pcount()&&Pi<distance.length; Pi++) 
				{
					int size = (int)(1/distance[Pi]);
					if(size>0) {brush.setColor(Color.CYAN);}else {brush.setColor(Color.RED);}
					size=Math.abs(size);
					int oldX = x;
					int oldY = y;
					x = (int)(PointTracer.onfrmX(Pi)*1000);
					y = (int)(PointTracer.onfrmY(Pi)*1000);
					size=10;
					brush.fillOval(x-size/2, y-size/2, size, size);
					brush.drawLine(oldX, oldY, x, y);
					System.out.println("fx:"+x);
					System.out.println("fy:"+y);
					System.out.println("distance:"+distance[Pi]);
				}
			}
		);
	}
	public void renderFramePoints() 
	{
		List<OpenCLDevice> devices = OpenCLDevice.listDevices(null);
        for (Device device: devices)
        {
           System.out.println(device.getShortDescription()+"; ID: "+device.getDeviceId()); 
        }
   	   Device device = Device.best();
       System.out.println(device.toString());
   	  //-----------------------------------------------------------------------------------
       int gs = PointTracer.pcount(); 
       System.out.println("Global thread size: "+gs);  
       com.aparapi.Range range = device.createRange(gs);
       try 
       {//===================================================
         	PointTracer.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
         	System.out.println("explicit: true");  
         	PointTracer.setExplicit(true);
         	PointTracer.get(PointTracer.getPointArray());
         	long  time1 = System.currentTimeMillis();
         	
         	for(int i=0; i<1; i++) 
         	{
         		PointTracer.execute(range);
         	}
         	PointTracer.get(PointTracer.getDistanceArray()); //Fetching value from Device to main memory!
         	PointTracer.get(PointTracer.getFramePointArray());
         	
         	System.out.println("Time taken for kenel execution in "+ PointTracer.getExecutionMode()+" mode is :"+(System.currentTimeMillis() - time1));
       }	
       catch(NullPointerException ne){ne.printStackTrace();}
       //====================================================
     //-------------------------
       distance = PointTracer.getDistanceArray();
       PointTracer.dispose();
	}
	
	public NPanel getSurface() 
	{
		return Surface;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
