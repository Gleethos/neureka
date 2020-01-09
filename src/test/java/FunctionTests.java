import neureka.function.Function;
import neureka.function.factory.assembly.FunctionBuilder;
import org.junit.Test;

public class FunctionTests {


    @Test
    public void testDerivatives(){

        Function f = FunctionBuilder.build("1/I[0]", false);
        assert f.derive(new double[]{2}, 0)==-0.25;
        assert f.activate(new double[]{2})==0.5;

        f = FunctionBuilder.build("I[0]+1/I[0]", false);
        assert f.activate(new double[]{2})==2.5;
        assert f.derive(new double[]{-1}, 0)==0.0;
        assert f.derive(new double[]{-3}, 0)==0.8888888888888888;
        assert f.derive(new double[]{0.2}, 0)==-23.999999999999996;

        f = FunctionBuilder.build("(I[0]+1/I[0])^-I[0]", false);
        assert f.activate(new double[]{1})==0.5;
        assert f.derive(new double[]{0.2}, 0)==-0.5217778675999797;

    }












}
