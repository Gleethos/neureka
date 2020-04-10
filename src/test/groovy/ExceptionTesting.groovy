import neureka.Tsr
import org.junit.Test

class ExceptionTesting
{
    @Test
    void setting_empty_subset_raises_proper_exception(){

        try {
            new Tsr([6, 6], -1)[[1..3], [1..3]] = new Tsr()
        } catch(Exception e){
            assert e.toString().contains("Provided tensor is empty!")
        }

    }

}
