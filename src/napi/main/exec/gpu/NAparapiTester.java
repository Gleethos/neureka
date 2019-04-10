package napi.main.exec.gpu;

 import napi.main.exec.gpu.wip.*;

import java.util.List;
import java.util.Random;

import com.aparapi.*;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.device.OpenCLDevice.DeviceSelector;

/**
 * @author Nepp Daniel - Vandermingo
 */
 
public class NAparapiTester  
{
    @SuppressWarnings("deprecation")
	public static void main(String [] args) throws Exception
    {
    	testConvector();
         //testNVConvector();
    }
    
    public static void testConvector() {
    	final int row = 500;//1024;
        final int com = 7000;//r;//4;
        final int col = 300;//r;//5;
        NAparapiMatMul ConvectorKernel = new NAparapiMatMul(row, com, col);
        
        //ap.printAll();

        List<OpenCLDevice> devices = OpenCLDevice.listDevices(null);
        for (Device device: devices){
           System.out.println(device.getShortDescription());
        }
        
        Device device = Device.best();
        System.out.println(device.toString());
        
        int gs = ConvectorKernel.convectionSize(); //Note if execution is AxB=C => r*col; If execution is AtxC=B => r*com!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    	//com.aparapi.Range range = device.createRange2D(r,col);
    	com.aparapi.Range range = device.createRange(gs);

        try 
        {//===================================================
        	
        	ConvectorKernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
        	//
        	//ap.setExplicit(true);
       
        	long  time1 = System.currentTimeMillis();
        	
        	for(int i=0; i<25; i++) {
        		ConvectorKernel.execute(range);
        	//ConvectorKernel.get(ConvectorKernel.getB()); //Fetching data from GPU to main memory! 
        	//ConvectorKernel.get(ConvectorKernel.getC());
        	}//ap.setA(new float [com *    r]);
        	
        	
        	System.out.println("Time taken for kenel execution in "+ ConvectorKernel.getExecutionMode()+" mode is :"+(System.currentTimeMillis() - time1));
        }	
        catch(NullPointerException ne){ne.printStackTrace();}
        //====================================================

        //ap.printResults();
        
        long time1 = System.currentTimeMillis();
        ConvectorKernel.cpuMatMulCalc();
        System.out.println("Time taken for kenel execution in Sequential CPU mode is :"+ (System.currentTimeMillis() - time1));     
        //ap.printAll();
        ConvectorKernel.compareResults();
        System.out.println("================================================\n");
        //ap.printResults();
        ConvectorKernel.dispose();
    }

    public static void testNVConvector() 
    {
    	int[] Bcomp = {3, 2, 3, 1, 4};
    	int[] AoBcomp = {-1,2,-1,1,  1,-2,2,   -1,4,   -2,2,-1};
    	int row = 3;
    	int col = 2;
    	
    	//all devices...
    	List<OpenCLDevice> devices = OpenCLDevice.listDevices(null);
         for (Device device: devices){
            System.out.println(device.getShortDescription()+"; ID: "+device.getDeviceId()); 
         }
 
    	NVCKernel ConvectorKernel = new NVCKernel(row, col, Bcomp, AoBcomp);
    	ConvectorKernel.printAll();
    	
    	Device device = Device.best();
        System.out.println(device.toString());
    	
    	//-----------------------------------------------------------------------------------
        int gs = ConvectorKernel.convectionSize(); //Note if execution is AxB=C => r*col; If execution is AtxC=B => r*com!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        System.out.println("Global thread size: "+gs);  
       	com.aparapi.Range range = device.createRange(gs);
        try 
        {//===================================================
          	ConvectorKernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
          	ConvectorKernel.setExplicit(true);
         
          	long  time1 = System.currentTimeMillis();
          	
          	for(int i=0; i<1; i++) 
          	{
          		ConvectorKernel.execute(range);
          	}
          	ConvectorKernel.get(ConvectorKernel.getB()); //Fetching data from GPU to main memory! 
          	ConvectorKernel.get(ConvectorKernel.getC());
          	
          	System.out.println("Time taken for kenel execution in "+ ConvectorKernel.getExecutionMode()+" mode is :"+(System.currentTimeMillis() - time1));
        }	
        catch(NullPointerException ne){ne.printStackTrace();}
        //====================================================

        ConvectorKernel.printResults();
        long time1 = System.currentTimeMillis();
        ConvectorKernel.cpuMatMulCalc();
        System.out.println("Time taken for kenel execution in Sequential CPU mode is :"+ (System.currentTimeMillis() - time1));     
        ConvectorKernel.printAll();
        ConvectorKernel.compareResults();
        System.out.println("================================================\n");
        ConvectorKernel.printResults();
        
        
        //-------------------------
        ConvectorKernel.dispose();
    }
}
