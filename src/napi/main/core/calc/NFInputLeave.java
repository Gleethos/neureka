package napi.main.core.calc;


public class NFInputLeave implements NVFunction
{
	int InputIndex;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public NVFunction newBuild(final String equation) 
    {
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
        	if(equation.charAt(i)=='j') 
        	{
        		NVFunction newCore = new NVFVariableLeave();
        		newCore = newCore.newBuild(equation);
        		return newCore;
        	}
            if (equation.charAt(i) <= '9' && equation.charAt(i) >= '0') 
            {
                number *= 10;
                number += Integer.parseInt(equation.charAt(i)+"");
            }
            
        }
        this.InputIndex = number;
        return (NVFunction)this;
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) 
    {
    	return input[InputIndex];
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, double[] bias) 
    {
    	double[] newInput = new double[input.length];
    	if(bias!=null) {for(int Ii=0; Ii<input.length; Ii++) {newInput[Ii]=input[Ii]+bias[Ii];}}
    	else {for(int Ii=0; Ii<input.length; Ii++) {newInput[Ii]=input[Ii];}}
    	return activate(newInput);
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input) 
    {
        return input[InputIndex];
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
  	public double derive(double[] input, double[] bias, int index) 
    {
    	if (index == this.InputIndex) {
            return 1.0;
        }
  			return 0;
  	}
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index) 
    {
        if (index == this.InputIndex) {
            return 1.0;
        }
        return 0.0;
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
	public double derive(double[] input, int index, int j) 
    {
		return derive(input,index);
	}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public boolean dependsOn(final int index) 
    {
        return this.InputIndex == index;
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String expression() 
    {
        return "I[" + this.InputIndex + "]";
    }
  
}
