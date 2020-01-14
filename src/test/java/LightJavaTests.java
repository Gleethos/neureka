import neureka.Tsr;
import org.junit.Test;

public class LightJavaTests {


    @Test
    public void testFlags(){
        Tsr t = new Tsr(6);
        assert !t.isOutsourced();
        t.setIsOutsourced(true);
        assert t.isOutsourced();
        assert  !t.is64() && !t.is32();
        assert t.value64()==null;
        assert  t.value32()==null;
        t.setIsOutsourced(false);
        assert t.value64()!=null;
        assert t.isVirtual();
        t = new Tsr(new int[]{2}, 5);
        assert !t.isOutsourced();
        t.setIsOutsourced(true);
        assert t.isOutsourced();
        assert  !t.is64() && !t.is32();
        assert t.value64()==null;
        assert  t.value32()==null;
        t.setIsOutsourced(false);
        assert t.value64()!=null;
        assert t.isVirtual();
    }



}
