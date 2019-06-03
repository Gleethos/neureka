package neureka.unit;

import neureka.main.core.NVNode;

import neureka.utility.NMessageFrame;
import java.math.BigInteger;

public class NVTesting {

    protected NMessageFrame Console;
    protected NMessageFrame ResultConsole;
    protected String bar = "[I]";

    protected NVTesting(NMessageFrame console, NMessageFrame resultConsole) {
        Console = console;
        ResultConsole = resultConsole;
    }

    protected void performForwardBackwardOn(NVNode[] Structure){
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadLatest();}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadLatest();  }
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadTrainableState(BigInteger.ZERO);}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().Backward(BigInteger.ZERO);}
    }

    protected void printNetStructView(NVNode[] Structure, NVNode head){
        for(int i=0; i<Structure.length; i++) {
            String expression = "null";
            if(Structure[i].asCore().getFunction()!=null) {
                expression = Structure[i].asCore().getFunction().expression();
            }
            Console.print(bar+"  Root ["+i+"]: ");
            if(Structure[i]==head) {Console.println(expression+"  => (head)");}
            else {
                Console.println(expression);
            }
        }
    }

    protected void printStart(String message){
        Console.println("");
        Console.println(bar+"  "+message);
        Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
    }

    protected boolean assertEqual(double result, double expected){
        if(result==expected) {
            Console.println(bar+"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            return true;
        } else {
            Console.println(bar+"  [result]:("+result+") =|= -> test failed!");
            return false;
        }
    }

    protected boolean assertEqual(String result, String expected){
        if(result.equals(expected)) {
            Console.println(bar+"  [result]:("+result+") "+((result.length()>22)?"\n"+bar+"    ==  "+"\n"+bar+" ":"==")+" [expected]:("+expected+") -> test successful.");
            return true;
        } else {
            Console.println(bar+"  [result]:("+result+") "+((result.length()>22)?"\n"+bar+"    =|=  "+"\n"+bar+" ":"=|=")+" [expected]:("+expected+") -> test failed!");
            return false;
        }
    }

    protected String stringified(int[] a){
        String result = "";
        for(int ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(short[] a){
        String result = "";
        for(short ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(double[] a){
        String result = "";
        for(double ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(double[][] a){
        String result = "";
        for(double[] ai : a) {
            result+="(";
            for(double aii : ai){
                result += aii+", ";
            }
            result+="), ";
        }
        return result;
    }
    protected String stringified(float[] a){
        String result = "";
        for(float ai : a) {
            result += ai+", ";
        }
        return result;
    }

    protected void print(String message){
        Console.print(message);
    }
    protected void println(String message){
        Console.println(message);
    }
    protected void printResult(String message){
        ResultConsole.print(message);
    }
    protected void printlnResult(String message){
        ResultConsole.println(message);
    }

}
