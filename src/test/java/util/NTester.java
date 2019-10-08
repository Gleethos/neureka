package util;
import neureka.core.Tsr;
import org.junit.Assert;
import neureka.frame.NMessageFrame;

public class NTester extends Assert
{
    private NMessageFrame _verbose_frame;
    private static NMessageFrame _result_frame;
    static {
        if(System.getProperty("os.name").toLowerCase().contains("windows")){
            _result_frame  = new NMessageFrame("[NEUREKA UNIT TEST]: results");
        }
    }
    protected static String BAR = "[|]";
    protected static String LINE = "--------------------------------------------------------------------------------------------";
    private int _positive_assertions = 0;
    private int _assertion_count = 0;

    private int _success = 0;
    private int _tests = 0;

    private String _session = "";

    public NTester(String name) {
        if(System.getProperty("os.name").toLowerCase().contains("windows")){
            _verbose_frame = new NMessageFrame("[NEUREKA UNIT TEST]:("+name+"): verbose results");
            printlnResult("\nT[ "+name+" ]:");
        }

    }

    public NTester(String name, boolean liveLog){
        if(liveLog && System.getProperty("os.name").toLowerCase().contains("windows")){
            _verbose_frame = new NMessageFrame(name+" - TEST PROCESS");
            printResult("\nT["+name+"]: ");
        }
    }

    public void closeWindows(){
        if(_verbose_frame !=null){
            _verbose_frame.close();
        }
        if(_result_frame !=null){
            _result_frame.close();
        }
    }

    public int testContains(String result, String[] expected, String description){
        printSessionStart(description);
        println(BAR + "  Tensor: "+result);
        println(BAR + "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    protected void printSessionStart(String message){
        _session = "";
        _tests++;
        _positive_assertions = 0;
        _assertion_count = 0;
        println("");
        println(BAR +"  "+message);
        println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        println(BAR + LINE);
    }
    protected int printSessionEnd(){
        _success += (_positive_assertions == _assertion_count)?1:0;
        println(BAR +"  "+((_positive_assertions >0)?"test successful!"+" "+ _positive_assertions :"test failed!"+" "+(_assertion_count + _positive_assertions))+"/"+ _assertion_count);
        println("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+ _success +"/"+ _tests);
        printResult((_positive_assertions == _assertion_count)?".":"E");
        bottom();
        return _positive_assertions;
    }

    protected boolean assertIsEqual(double result, double expected){
        _assertion_count++;
        if(result==expected) {
            println(BAR +"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            return true;
        } else {
            println(BAR +"  [result]:("+result+") =|= [expected]:("+expected+") -> test failed!");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            return false;
        }
    }

    protected boolean assertStringContains(String name, String result, String expected){
        _assertion_count++;
        if(result.contains(expected)) {
            println(BAR +"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+ BAR +"    contains   -> test successful!"+"\n"+ BAR +" ":"contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertEquals(true, true);
            return true;
        } else {
            println(BAR +"  ["+name+"]:("+result+") "
                +((result.length()>22)
                    ?"\n"+ BAR +"    not contains   -> test failed!"+"\n"+ BAR +" "
                    :"not contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            assertEquals(_session, false, true);
            return false;
        }
    }

    protected boolean assertIsEqual(String name, String result, String expected){
        _assertion_count++;
        if(result.equals(expected)) {
            println(BAR +"  ["+name+"]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    ==   -> test successful!"+"\n"+ BAR +" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?""
                    :" -> test successful!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertEquals(true, true);
            return true;
        } else {
            println(BAR +"  ["+name+"]:("+result+") "
                    +((result.length()>22)
                    ?"\n"+ BAR +"    =|=   -> test failed!"+"\n"+ BAR +" "
                    :"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            assertEquals(_session, false, true);
            return false;
        }
    }

    protected boolean assertIsEqual(String result, String expected){
        _assertion_count++;
        if(result.equals(expected)) {
            println(BAR +"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    ==  "+"\n"+ BAR +" "
                    :"==")+" [expected]:("+expected+") -> test successful.");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertEquals(true, true);
            return true;
        } else {
            println(BAR +"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    =|=  "+"\n"+ BAR +" "
                    :"=|=")+" [expected]:("+expected+") -> test failed!");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            assertEquals(_session, false, true);
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
            result += Tsr.factory.util.formatFP(ai)+", ";
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
        _session +=message;
        if(_verbose_frame !=null){
            _verbose_frame.print(message);
        }
    }
    protected void println(String message){
        _session +=message+"\n";
        if(_verbose_frame !=null){
            _verbose_frame.println(message);
        }
    }
    protected void printResult(String message){
        if(_result_frame !=null){
            _result_frame.print(message);
        }
    }
    protected void printlnResult(String message){
        if(_result_frame !=null){
            _result_frame.println(message);
        }
    }

    protected void bottom(){
        if(_verbose_frame !=null && _result_frame !=null){
            _verbose_frame.bottom();
            _result_frame.bottom();
        }
    }

}
