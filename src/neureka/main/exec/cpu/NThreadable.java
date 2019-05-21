
package neureka.main.exec.cpu;

import java.math.BigInteger;

import neureka.main.exec.NVExecutable;


public class NThreadable implements Runnable {

    private NVExecutable Executable;
    private int ID;
    private boolean isBackpropagating;

    //Construction functions:
//============================================================================================================================================================================================	
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
    //In use:
    public NThreadable(NVExecutable executable, int threadID, boolean backProp) {
        Executable = executable;
        ID = threadID;
        isBackpropagating = backProp;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public void run() {
        if (isBackpropagating) {
            Executable.Backward(BigInteger.ZERO);
        } else {//u.startForwardpropagation();
            Executable.forward();
            //memorize(Core.activationOf(Core.weightedConvectionOf(Core.publicized(Core.inputSignalIf(Core.forwardCondition())))));
            //Memorize activation of weighted convection of publicized input signal if forward condition is true!
        }
    }

    public void setIsBackpropagating(boolean value) {
        isBackpropagating = value;
    }

    public boolean isBackpropagating() {
        return isBackpropagating;
    }

    public NVExecutable getExecutable() {
        return Executable;
    }

    public int getID() {
        return ID;
    }
}