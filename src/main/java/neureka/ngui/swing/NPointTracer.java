package neureka.ngui.swing;

import com.aparapi.Kernel;

public class NPointTracer extends Kernel
{
	/**
	 * 	Frame plane definition:
	 * 
	 *  (x-fpX)*fvX + (y-fpY)*fvY + (z-fpZ)*fvZ = 0
	 * 
	 * 	Ray:
	 * 
	 *  x = pX + s*(vpX-pX)
	 *  y = pY + s*(vpY-pY)
	 *  z = pZ + s*(vpZ-pZ)
	 * 
	 *  fx = ynX*u + xnX*k + fpX;
	 *  fy = ynY*u + xnY*k + fpY;
	 *  fz = ynZ*u + xnZ*k + fpZ;
	 *  
	 *  k = (fx - ynX*u - fpX)/xnX
	 *  u = (fy - xnY*k - fpY)/ynY
	 *  
	 *  k = (fx - ynX*( (fy - xnY*k - fpY)/ynY ) - fpX)/xnX
	 *  k*xnX = (fx - ynX*fy/ynY - ynX*ynY*k/ynY - fpY/ynY - fpX);
	 *  k*xnX + ynX*ynY*k/ynY = fx - ynX*fy/ynY - fpY/ynY - fpX
	 *  k = (fx - ynX*fy/ynY - fpY/ynY - fpX)/(xnX + (ynX*ynY)/ynY)
	 *  
	 * */
	//CAMERA:
	float vpX, vpY, vpZ;//VIEW POINT
	float fpX, fpY, fpZ;//FRAME CENTER
	float fvX, fvY, fvZ;//FRAME VECTOR
	
	float ynX, ynY, ynZ;
	float xnX, xnY, xnZ;
	
	public void setViewPointX(float value) {vpX=value;}
	public void setViewPointY(float value) {vpY=value;}
	public void setViewPointZ(float value) {vpZ=value;}
	public void setFramePointX(float value) {fpX=value;}
	public void setFramePointY(float value) {fpY=value;}
	public void setFramePointZ(float value) {fpZ=value;}
	public void setFrameVecX(float value) {fvX=value;}
	public void setFrameVecY(float value) {fvY=value;}
	public void setFrameVecZ(float value) {fvZ=value;}
	//------------------------------------------------------------------------
	float[] point;//VERTECIES:
	public float[] getPointArray() {return point;}
	public void    setPointArray(float[] array) {point = array;}
	public void  setPointX(int p_i, float value) {point[3*p_i]=value;}
	public void  setPointY(int p_i, float value) {point[3*p_i+1]=value;}
	public void  setPointZ(int p_i, float value) {point[3*p_i+2]=value;}
	public float pointX(int p_i) {return point[3*p_i];}
	public float pointY(int p_i) {return point[3*p_i+1];}
	public float pointZ(int p_i) {return point[3*p_i+2];}
	public int   pcount() {return point.length/3;}
	//-----------------------------------------------------------------------
	float[] onfrm;//FRAME POINTS:
	public float[] getFramePointArray() {return onfrm;}
	public void    setFramePointArray(float[] array) {onfrm = array;}
	public void  setOnfrmX(int p_i, float value) {onfrm[2*p_i]=value;}
	public void  setOnfrmY(int p_i, float value) {onfrm[2*p_i+1]=value;}
	public float onfrmX(int p_i) {return onfrm[2*p_i];}
	public float onfrmY(int p_i) {return onfrm[2*p_i+1];}
	//-----------------------------------------------------------------------
	private float[] dstnc;//VIEW POINT TO POINTS DISTANCES:
	public float[] getDistanceArray() {return dstnc;}
	public void  setDstncOf(int p_i, float value) {dstnc[p_i]=value;}
	public float dstncOf(int p_i) {return dstnc[p_i];}
	//=======================================================================
	public NPointTracer(float[] points)
	{//-------------------------------
		point = points;
		vpX=0; vpY=0; vpZ=0;
		fpX=-5; fpY=0; fpZ=0;
		fvX=-5; fvY=0; fvZ=0;
		dstnc = new float[point.length/3];
		onfrm = new float[(point.length/3)*2];
		ynX=0; ynY=0; ynZ=1;
		xnX=0; xnY=1; xnZ=0;
	}//-------------------------------
	
	@Override
	public void run() 
	{
		int p_i = getGlobalId();
		
		float pX = pointX(p_i);
		float pY = pointY(p_i);
		float pZ = pointZ(p_i);
		
		float vX = vpX-pX;
		float vY = vpY-pY;
		float vZ = vpZ-pZ;
		
		setDstncOf(p_i, (float) Math.pow((vX*vX + vY*vY),0.5));
		
		float s = (-pX*fpX-pY*fpY-pZ*fpZ)/(vX + vY + vZ);
		float fx = pX + s*(vX);
		float fy = pY + s*(vY);
		float fz = pZ + s*(vZ);
	
		float k = (fx - ynX*fy/ynY - fpY/ynY - fpX)/(xnX + (ynX*ynY)/ynY);
		float u = (fy - xnY*k - fpY)/ynY;	
		onfrm[2*p_i] = k;
		onfrm[2*p_i+1] = u;
		
	}
	/** Formular:
	 * 
	 *  (x-fpX)*fvX + (y-fpY)*fvY + (z-fpZ)*fvZ = 0
	 *  x = pX + s*(vpX-pX)
	 *  y = pY + s*(vpY-pY)
	 *  z = pZ + s*(vpZ-pZ)
	 *  s = (-pX*fpX-pY*fpY-pZ*fpZ)/((vpX-pX)+(vpY-pY)+(vpZ-pZ))
	 *  
	 *  fx = ynX*u + xnX*k + fpX;
	 *  fy = ynY*u + xnY*k + fpY;
	 *  fz = ynZ*u + xnZ*k + fpZ;
	 *  
	 *  k = (fx - ynX*u - fpX)/xnX
	 *  u = (fy - xnY*k - fpY)/ynY
	 *  
	 *  k = (fx - ynX*( (fy - xnY*k - fpY)/ynY ) - fpX)/xnX
	 *  k*xnX = (fx - ynX*fy/ynY - ynX*ynY*k/ynY - fpY/ynY - fpX);
	 *  k*xnX + ynX*ynY*k/ynY = fx - ynX*fy/ynY - fpY/ynY - fpX
	 *  k = (fx - ynX*fy/ynY - fpY/ynY - fpX)/(xnX + (ynX*ynY)/ynY)
	 *  
	 *  
	 * */

}
