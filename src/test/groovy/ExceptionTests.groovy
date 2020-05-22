import neureka.Tsr
import org.junit.Test

class ExceptionTests
{
    @Test
    void setting_empty_subset_raises_proper_exception(){

        try {
            new Tsr([6, 6], -1)[[1..3], [1..3]] = new Tsr()
            assert false
        } catch(Exception e){
            assert e.toString().contains("Provided tensor is empty!")
        }

    }

    @Test
    void constructor_exception(){
        try {
            new Tsr(new Scanner(System.in))
            assert false
        } catch(Exception e){
            assert e.toString().contains(
                    "IllegalArgumentException: Cannot create tensor from argument of type 'java.util.Scanner'!"
            )
        }

    }

}
