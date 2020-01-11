import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
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

        f = FunctionBuilder.build("(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)", false);
        assert f.activate(new double[]{3})==3.049021713079475;
        assert f.activate(new double[]{2.5})==3.507365283517986;
        assert f.activate(new double[]{0})==0.2;

        assert f.derive(new double[]{0}, 0)==1.1;
        assert f.derive(new double[]{0.5}, 0)==0.646867884000033;
        assert f.derive(new double[]{1.6}, 0)==-0.00697440343353687;
        assert f.derive(new double[]{-4}, 0)==3.9174193383745917;

        f = FunctionBuilder.build("sum((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))", false);
        double[] inputs = {0.0, 0.5, 1.6, -4.0};
        double[] expected = {1.1, 0.646867884000033, -0.00697440343353687, 3.9174193383745917};
        for(int i=0; i<expected.length; i++){
            assert f.derive(inputs, i)==expected[i];
        }

    }












}
