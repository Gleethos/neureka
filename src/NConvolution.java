
public class NConvolution {

	int mainFrameHight, mainFrameWidth, mainFrameLength, mainFrameDepth;
	int mainFrameNCount;
	
	int referenceFrameHight, referenceFrameWidth, referenceFrameLength, referenceFrameDepth;
	int referenceFrameNCount;
	
	int convolutionFrameHight, convolutionFrameWidth, convolutionFrameLength, convolutionFrameDepth;
	int convolutionFrameNCount;
	
	boolean[][][][] convolutionFilter; 
	boolean[][]     linearConvolutionFilter;
	
//Construction functions:	
//============================================================================================================================================================================================	
	NConvolution(int hight, int width, int length, int depth){setFrameDimensionsTo(hight, width, length, depth);}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	NConvolution(int primaryHight, int primaryWidth, int primaryLength, int primaryDepth, int referenceHight, int referenceWidth, int referenceLength, int referenceDepth)
	{setFrameDimensionsTo(primaryHight, primaryWidth, primaryLength, primaryDepth); setConvolutionReferenceFrameDimensionsTo(referenceHight, referenceWidth, referenceLength, referenceDepth);
	 createConvolutionFilterSets();}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	public void setFrameDimensionsTo                    (int hight, int width, int length, int depth)
	{mainFrameHight=hight; mainFrameWidth=width; mainFrameLength=length; mainFrameDepth=depth; mainFrameNCount=hight*width*depth;
	 System.out.println("main frame height: "+mainFrameHight+" main frame width: "+mainFrameWidth+" main frame length: "+mainFrameLength+" main frame depth: "+mainFrameDepth);}
	
	public void setConvolutionReferenceFrameDimensionsTo(int hight, int width, int length, int depth)
	{referenceFrameHight=hight; referenceFrameWidth=width; referenceFrameLength=length; referenceFrameDepth=depth; referenceFrameNCount=hight*width*depth;
	 System.out.println("reference frame height: "+referenceFrameHight+" reference frame width: "+referenceFrameWidth+" main frame length: "+mainFrameLength+" main frame depth: "+mainFrameDepth);}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	

	public boolean[][] getLinearConvolutionFilter()					   {return linearConvolutionFilter;}
	public boolean[]   getLinearConvolutionFilterOfNeuron(int neuronID){return linearConvolutionFilter[neuronID];}
	
	public boolean[][][][] getConvolutionFilter()                    {return convolutionFilter;}
	public boolean[][][]   getConvolutionFilterOfNeuron(int neuronID){return convolutionFilter[neuronID];}
	
	//CONVOLUTION:
	//=============================================================================================================================================================================================
	public void createConvolutionFilterSets()
	{convolutionFrameHight=referenceFrameHight-mainFrameHight;
	 convolutionFrameWidth=referenceFrameWidth-mainFrameWidth;
	 convolutionFrameLength=referenceFrameLength-mainFrameLength;
	 convolutionFrameDepth=referenceFrameDepth-mainFrameDepth;
	 System.out.println("convolution frame hight: "+convolutionFrameHight+"convolution frame width: "+convolutionFrameWidth+"convolution frame length: "+convolutionFrameLength+"convolution frame depth: "+convolutionFrameDepth);	 
	 convolutionFilter = new boolean[mainFrameNCount][referenceFrameHight][referenceFrameWidth][referenceFrameDepth];
	 
	 //Counting through the mainFrame:
	 int mainNCounter=0;
	 for(                int mainFrameHightCounter=0; mainFrameHightCounter<mainFrameHight; mainFrameHightCounter++)
	     {for(           int mainFrameWidthCounter=0; mainFrameWidthCounter<mainFrameWidth; mainFrameWidthCounter++)  
	          {for(      int mainFrameLengthCounter=0; mainFrameLengthCounter<mainFrameLength; mainFrameLengthCounter++)  
	               {for( int mainFrameDepthCounter=0; mainFrameDepthCounter<mainFrameDepth; mainFrameDepthCounter++)    
	                    {//Counting through the referenceFrame: 
	        	         //-----------------------------------------------------------------------------------------------------------------------------------------
				         int referenceNCounter=0;
	        	  	     for(                int referenceFrameHightCounter=0; referenceFrameHightCounter<referenceFrameHight; referenceFrameHightCounter++)
				             {for(           int referenceFrameWidthCounter=0; referenceFrameWidthCounter<referenceFrameWidth; referenceFrameWidthCounter++)  
				                  {for(      int referenceFrameLengthCounter=0; referenceFrameLengthCounter<referenceFrameLength; referenceFrameLengthCounter++) 
				                       {for( int referenceFrameDepthCounter=0; referenceFrameDepthCounter<referenceFrameDepth; referenceFrameDepthCounter++)    
				                            {
				                	           if(referenceFrameHightCounter>mainFrameHightCounter&referenceFrameHightCounter<(referenceFrameHight-mainFrameHightCounter))
							                      {convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=true;  
							                       linearConvolutionFilter[mainNCounter][referenceNCounter]=true;}
					                          else{convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=false; 
					                               linearConvolutionFilter[mainNCounter][referenceNCounter]=false;};
							 
						                       if(referenceFrameWidthCounter>mainFrameWidthCounter&referenceFrameWidthCounter<(referenceFrameWidth-mainFrameWidthCounter))
						                          {convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=true; 
						                           linearConvolutionFilter[mainNCounter][referenceNCounter]=true;}
						                      else{convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=false;
						                           linearConvolutionFilter[mainNCounter][referenceNCounter]=false;};
							 
						                       if(referenceFrameLengthCounter>mainFrameLengthCounter&referenceFrameLengthCounter<(referenceFrameLength-mainFrameLengthCounter))
					                              {convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=true; 
					                               linearConvolutionFilter[mainNCounter][referenceNCounter]=true;}
					                          else{convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=false;
					                               linearConvolutionFilter[mainNCounter][referenceNCounter]=false;}; 
					                      
					                           if(referenceFrameDepthCounter>mainFrameDepthCounter&referenceFrameDepthCounter<(referenceFrameDepth-mainFrameDepthCounter))
				                                  {convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=true; 
				                                   linearConvolutionFilter[mainNCounter][referenceNCounter]=true;}
				                              else{convolutionFilter[mainNCounter][referenceFrameHightCounter][referenceFrameWidthCounter][referenceFrameDepthCounter]=false;
				                                   linearConvolutionFilter[mainNCounter][referenceNCounter]=false;};  
						                     referenceNCounter++;
				                            }
				                       }
				                  }
				             }
	        	  	     	//-----------------------------------------------------------------------------------------------------------------------------------------
	        	  	     	mainNCounter++;  
	                    } 
	               }	 
	          }
	     }
	}
//=============================================================================================================================================================================================	

}
