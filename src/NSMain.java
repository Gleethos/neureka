import neureka.NSubstrate;
import neureka.main.core.NVUtility;
import neureka.main.core.base.NVNeuron;
import neureka.unit.NVTester;
import neureka.ngui.NWindow;
import neureka.utility.NConsole;

public class NSMain {

	static NWindow testWindow;
	static GridGame game;
    
	public static void main(String[] args) throws InterruptedException
	{

		NConsole console = new NConsole("Huuui", 1000);
		//NSoundPlayer player = new NSoundPlayer();
		//		player.playSound("droplet shatter.wav");
				//NAparapi test = new NAparapi();
				//test.start();
		
		//System.out.println("NTest");
        //testWindow = new NWindow();
        
		//double[] inputLine = {1, 0.2};
       // NUnitCoreCalculator Calc = new NCalculator();
		//Calc = Calc.build("prod(tanh(Ij))");
		//System.out.println(Calc.equation());
		//System.out.println(Calc.activate(inputLine));
		//System.out.println(Calc.derive(inputLine,0));
        /*
		//======================================
		// Open a database connection
        // (create a new database if it doesn't exist yet):
        EntityManagerFactory emf =
            Persistence.createEntityManagerFactory("$objectdb/db/Cores.odb");
        EntityManager em = emf.createEntityManager();

        javax.persistence.Query myQuery 
        = em.createQuery("SELECT p FROM NUnitCore p WHERE p.ID = 1", NUnitCore.class);
        java.util.List<NUnitCore> list = myQuery.getResultList();
        
        System.out.println("List size: "+list.size()+"; List: "+list);
        // Store 1000 Point objects in the database:
        for (int i = 1; i < 10; i++) {
        	String statement = "SELECT p FROM NUnitCore p WHERE p.ID = "+Integer.toString(i);
        	System.out.println(statement);
        	myQuery = em.createQuery(statement, NUnitCore.class);
        	em.getTransaction().begin();
            NUnitCore p = new NUnitCore(false, false, i, i);
            
            list = myQuery.getResultList();
            System.out.println("List size: "+list.size()+"; List: "+list);
            if(list.size()==0) 
            {System.out.println("Persiting now!");em.persist(p);}
            //else{em.clear();}
            
           em.getTransaction().commit(); 
        }
       
        // Find the number of Point objects in the database:
        javax.persistence.Query q1 = em.createQuery("SELECT COUNT(p) FROM NUnitCore p");
        System.out.println("Total Points: " + q1.getSingleResult());

        // Find the average X value:
        javax.persistence.Query q2 = em.createQuery("SELECT AVG(p.ID) FROM NUnitCore p");
        System.out.println("Average X: " + q2.getSingleResult());

        // Retrieve all the Point objects from the database:
        TypedQuery<NUnitCore> query =
            em.createQuery("SELECT p FROM NUnitCore p", NUnitCore.class);
        java.util.List<NUnitCore> results = query.getResultList();
        for (NUnitCore p : results) {
            System.out.println(p);
        }

        // Close the database connection:
        em.close();
        emf.close();
		
		//======================================
		 * 
		 */

		//NNeuronCompartpentCreator comp = new NNeuronCompartpentCreator();
		//comp.test();
		
		//NTester tester1 = new NTester();
		//tester1.Test();
		
	    
		
		int a = 673468345;
		int b = 874749349;
	    a^=b;b^=a;a^=b;
	    System.out.println(a+"  -  "+b);
	    
	    NVUtility util = new NVUtility();	    
	    System.out.println(Math.pow(Math.E, Math.PI)*Math.pow(Math.PI, Math.E)*Math.PI*Math.E);

	    for(int i=-100000; i<100001; i++) {
	    	System.out.println(i+" => "+util.getDoubleOf(i));
	    }
	    
		
        NVNeuron[] list = new NVNeuron[1];
        for (int i = 0; i < list.length; i++) {
            list[i] = new NVNeuron(1);
        }
        // Get the Java runtime
        Runtime runtime = Runtime.getRuntime();
        // Run the garbage collector
        runtime.gc();
        // Calculate the used memory
        long memory = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("All used memory: " + memory+" bytes; "+bytesToMegabytes(memory)+" megabytes;");
        memory -=556152;
        System.out.println("Actual used memory: "+memory+" bytes; " +bytesToMegabytes(memory)+" megabytes;");
        memory-=8;
        System.out.println("Used memory minus 8: "+memory+" bytes; "  + bytesToMegabytes(memory)+" megabytes;");
		
        NVTester tester2 = new NVTester();
	     tester2.Test();

		//NSubstrate.execute("NSubstrate.printRealms('test')");
		/*
		JFrame window = new JFrame("imba");
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setSize(400, 400);
		window.setLocationRelativeTo(null);
		window.setResizable(true);
		//window.setLayout(null);
		window.addKeyListener(null);
		window.setFocusable(true);
		window.setUndecorated(false);
		
		N3DSpace plot = new N3DSpace(400);
		window.getContentPane().e_add(plot.getSurface());
		window.setVisible(true);
		
		Thread.sleep(1000);
		plot.renderFramePoints();
		 */
		
        //testWindow = new NWindow();

		//NCluster cluster = new NCluster();
		//cluster.Test();
		//testNetwork();
		//
		//Array test;
		
		//NDoubleArray Storage = new NDoubleArray(4);
		//for(int i=0; i<Storage.length(); i++) 
		//{Storage.Array[i].e_set(3); System.out.println(Array.e_get(i));}
		

	}

	private static final long MEGABYTE = 1024L * 1024L;
	public static long bytesToMegabytes(long bytes) {
        return bytes / MEGABYTE;
    }
	
	
	
}
