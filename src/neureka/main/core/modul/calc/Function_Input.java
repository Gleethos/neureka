package neureka.main.core.modul.calc;


import neureka.main.core.base.data.T;

public class Function_Input implements Function
{
	int InputIndex;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public boolean isFlat(){
        return  false;
    }
    @Override
    public Function newBuild(final String equation)
    {
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
        	if(equation.charAt(i)=='j') 
        	{
        		Function newCore = new Function_Variable();
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
        return this;
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) 
    {
    	return input[InputIndex];
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input) 
    {
        return input[InputIndex];
    }
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index) 
    {
        return (index == this.InputIndex)?1:0;
    }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
	public double derive(double[] input, int index, int j) 
    {
		return derive(input,index);
	}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString()
    {
        return "I[" + this.InputIndex + "]";
    }

    @Override
    public T activate(T[] input, int j) {
        return  input[InputIndex];
    }

    @Override
    public T activate(T[] input) {
        return input[InputIndex];
    }

    @Override
    public T derive(T[] input, int index, int j) {
        return derive(input, index);
    }

    @Override
    public T derive(T[] input, int index) {
        return (index==this.InputIndex)
                ?T.factory.newTensor(1, input[0].shape())
                :T.factory.newTensor(0, input[0].shape());
    }

}
