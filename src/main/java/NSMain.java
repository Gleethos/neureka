
import neureka.ngui.swing.NWindow;

public class NSMain {

	public static void main(String[] args) throws Exception
	{

		//for(int i=0; i<300; i++){
		//	System.out.println(i+" -=> "+(char)i);
		//}
		//System.out.println("");

		// (char)187 //>>
		// (char)171 //<<

	    /**
         *  3, 5, 4, 2,
         *
         *  1, 3, 15, 60
         *
         *  1, 0, -2, -2,
         *
         *
         * */

        //NSurface app = new NSurface();
        //app.go();

        //TestWindow tw = new TestWindow();

        //tw.go(null);

         //NWindow window = new NWindow();

        //while(false||new Random().nextInt()!=32487){
        //}
		//NConsoleFrame console = new NConsoleFrame("Huuui", 1000);
		//NSoundPlayer player = new NSoundPlayer();
		//		player.playSound("droplet shatter.wav");
				//NAparapi test = new NAparapi();
				//test.start();
		
		//System.out.println("NTester");
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
       
        // Find the number f Point objects in the database:
        javax.persistence.Query q1 = em.createQuery("SELECT COUNT(p) FROM NUnitCore p");
        System.out.println("Total Points: " + q1.getSingleResult());

        // Find the average X _value:
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
		
	    
		
		//int a = 673468345;
		//int b = 874749349;
	    //a^=b;b^=a;a^=b;
	    //System.out.println(a+"  -  "+b);

	    //DataHelper util = new DataHelper();
	    //System.out.println(Math.pow(Math.E, Math.PI)*Math.pow(Math.PI, Math.E)*Math.PI*Math.E);
		//double sum = 0;
	    //for(int i=-1000000; i<200000; i++) {
	    //	double v = util.getDoubleOf(i);
	    //	sum+=v;
	    //	System.out.println(i+" => "+v);
	    //	System.out.println("    --===> "+sum);
	    //}
	    //if(true){
	    //	return;
		//}
	    //
		//
        //Object[] list = new Object[1];
        //for (int i = 0; i < list.length; i++) {
        //    list[i] = new Object();
        //}
        //// Get the Java runtime
        //Runtime runtime = Runtime.getRuntime();
        //// Run the garbage collector
        //runtime.gc();
        //// Calculate the used memory
        //long memory = runtime.totalMemory() - runtime.freeMemory();
        //System.out.println("All used memory: " + memory+" bytes; "+bytesToMegabytes(memory)+" megabytes;");
        //memory -=556152;
        //System.out.println("Actual used memory: "+memory+" bytes; " +bytesToMegabytes(memory)+" megabytes;");
        //memory-=8;
        //System.out.println("Used memory minus 8: "+memory+" bytes; "  + bytesToMegabytes(memory)+" megabytes;");

        //NeurekaTest tester2 = new NeurekaTest();
        //tester2.Test();



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
		window.getContentPane().addInto(plot.getSurface());
		window.setVisible(true);
		
		Thread.sleep(1000);
		plot.renderFramePoints();
		 */
		
        //testWindow = new NWindow();

		//NCluster cluster = new NCluster();
		//cluster.Test();
		//testNetwork();
		//
		//DataHelper test;
		
		//NDoubleArray Storage = new NDoubleArray(4);
		//for(int i=0; i<Storage.length(); i++) 
		//{Storage.DataHelper[i].setInto(3); System.out.println(DataHelper.getFrom(i));}
		//NNetworkWindow nw = new NNetworkWindow();
		//nw.start(new Stage());

	}

	private static final long MEGABYTE = 1024L * 1024L;
	public static long bytesToMegabytes(long bytes) {
        return bytes / MEGABYTE;
    }
	
	
	
}
