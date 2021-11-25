package st.tests

import neureka.Neureka
import neureka.devices.host.CPU

class StandaloneBenchmark {

	static void main(String[] args) {

		Neureka.get().reset()
		// Configure printing of tensors to be more compact:
		Neureka.get().settings().view().tensors = "dgc"

		def configuration = [ "iterations":1, "sample_size":20, "difficulty":15, "intensifier":0 ]

		def session = new GroovyShell().evaluate(
											new File("src/test/resources/benchmark.groovy").text
											// Only in test context:
											//Utility.readResource("benchmark.groovy", this)
										)

		String hash = ""
		String expected = "56b2eb74955e49cd777469c7dad0536e"

		session(
				configuration, null, CPU.get(),
				tsr -> {
					hash = (hash+tsr.toString()).md5()
				}
		)

		assert hash == expected

	}

}
