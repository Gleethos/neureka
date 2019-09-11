
import neureka.unit.NeurekaTest;
import org.junit.Test;
import com.aparapi.Kernel;

public class TestTest {

	@Test
	public void test1() throws InterruptedException {
		System.out.println("Test 1 works");
		NeurekaTest tester2 = new NeurekaTest();
		tester2.Test();
		Thread.sleep(10000);
	}

}