package util;
import org.junit.Assert;
import neureka.frame.NMessageFrame;

public class NTester extends Assert{

    private NMessageFrame Console;
    private NMessageFrame ResultConsole;
    protected String bar = "[|]";
    protected String line = "--------------------------------------------------------------------------------------------";
    private int positiveAssertions = 0;
    private int assertionCount = 0;

    private int success = 0;
    private int tests = 0;

    private String session = "";

    public NTester(String name) {
        if(System.getProperty("os.name").toLowerCase().contains("windows")){
            this.Console = new NMessageFrame(name+" - TEST PROCESS");
            this.ResultConsole = new NMessageFrame(name+" - TEST RESULT");
        }

    }

    public NTester(String name, boolean liveLog){
        if(liveLog && System.getProperty("os.name").toLowerCase().contains("windows")){
            this.Console = new NMessageFrame(name+" - TEST PROCESS");
            this.ResultConsole = new NMessageFrame(name+" - TEST RESULT");
        }
    }

    public int testContains(String result, String[] expected, String description){
        printSessionStart(description);
        println(bar+"  Tensor: "+result);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    protected void printSessionStart(String message){
        this.session = "";
        tests++;
        this.positiveAssertions = 0;
        this.assertionCount = 0;
        println("");
        println(bar+"  "+message);
        println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        println(bar+line);
        printlnResult("");
        printlnResult(bar+"  "+message);
        printlnResult("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        printlnResult(bar);
    }
    protected int printSessionEnd(){
        success += (positiveAssertions== assertionCount)?1:0;
        println(bar+"  "+((positiveAssertions >0)?"test successful!"+" "+ positiveAssertions :"test failed!"+" "+(assertionCount + positiveAssertions))+"/"+ assertionCount);
        println("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+success+"/"+tests);
        printlnResult(bar+"  "+((positiveAssertions >0)?"test successful!"+" "+ positiveAssertions :"test failed!"+" "+(assertionCount + positiveAssertions))+"/"+ assertionCount);
        printlnResult("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+success+"/"+tests);
        bottom();
        return positiveAssertions;
    }

    protected boolean assertIsEqual(double result, double expected){
        assertionCount++;
        if(result==expected) {
            println(bar+"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") =|= [expected]:("+expected+") -> test failed!");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertStringContains(String name, String result, String expected){
        assertionCount++;
        if(result.contains(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+bar+"    contains   -> test successful!"+"\n"+bar+" ":"contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            assertEquals(true, true);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                +((result.length()>22)
                    ?"\n"+bar+"    not contains   -> test failed!"+"\n"+bar+" "
                    :"not contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            assertEquals(session, false, true);
            return false;
        }
    }

    protected boolean assertIsEqual(String name, String result, String expected){
        assertionCount++;
        if(result.equals(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==   -> test successful!"+"\n"+bar+" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?""
                    :" -> test successful!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            assertEquals(true, true);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                    +((result.length()>22)
                    ?"\n"+bar+"    =|=   -> test failed!"+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            assertEquals(session, false, true);
            return false;
        }
    }

    protected boolean assertIsEqual(String result, String expected){
        assertionCount++;
        if(result.equals(expected)) {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==  "+"\n"+bar+" "
                    :"==")+" [expected]:("+expected+") -> test successful.");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            assertEquals(true, true);
            return true;
        } else {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    =|=  "+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+") -> test failed!");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            assertEquals(session, false, true);
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
    public String stringified(double[] a){
        if(a==null){
            return "null";
        }
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
        session+=message;
        if(Console!=null){
            Console.print(message);
        }
    }
    public void println(String message){
        session+=message+"\n";
        if(Console!=null){
            Console.println(message);
        }
    }
    protected void printResult(String message){
        if(ResultConsole!=null){
            ResultConsole.print(message);
        }
    }
    protected void printlnResult(String message){
        if(ResultConsole!=null){
            ResultConsole.println(message);
        }
    }

    protected void bottom(){
        if(Console!=null && ResultConsole!=null){
            Console.bottom();
            ResultConsole.bottom();
        }
    }

}
