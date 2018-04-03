/**
 * 
 */
package edu.wisc.cs.will.ILP.Regression;

import java.util.List;

import edu.wisc.cs.will.Boosting.Common.RunBoostedModels;
import edu.wisc.cs.will.Boosting.EM.HiddenLiteralState;
import edu.wisc.cs.will.Boosting.MLN.RunBoostedRLR;
import edu.wisc.cs.will.Boosting.RDN.RegressionRDNExample;
import edu.wisc.cs.will.Boosting.RDN.WILLSetup;
import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.DataSetUtils.RegressionExample;
import edu.wisc.cs.will.FOPC.Literal;
import edu.wisc.cs.will.ILP.LearnOneClause;
import edu.wisc.cs.will.ILP.SingleClauseNode;
import edu.wisc.cs.will.Utils.ProbDistribution;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.stdAIsearch.SearchInterrupted;

/**
 * @author tkhot
 *
 */
public class RegressionInfoHolderForMLN extends RegressionInfoHolderForRDN {
	public static Double totalCount; 
	
	public RegressionInfoHolderForMLN() {
		super();
	}

//	
//	
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtSuccess()
//	 */
//	@Override
//	public double weightedVarianceAtSuccess() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtFailure()
//	 */
//	@Override
//	public double weightedVarianceAtFailure() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtSuccess()
//	 */
//	@Override
//	public double totalExampleWeightAtSuccess() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtFailure()
//	 */
//	@Override
//	public double totalExampleWeightAtFailure() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtSuccess()
//	 */
//	@Override
//	public double meanAtSuccess() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtFailure()
//	 */
//	@Override
//	public double meanAtFailure() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	/* (non-Javadoc)
//	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#addFailureStats(edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder)
//	 */
//	@Override
//	public RegressionInfoHolder addFailureStats(RegressionInfoHolder addThis) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//
//	@Override
//	public void addFailureExample(Example eg, long numGrndg, double weight) {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public double variance() {
//		// TODO Auto-generated method stub
//		return 0;
//	}

	@Override
	public void populateExamples(LearnOneClause task, SingleClauseNode caller) throws SearchInterrupted {
		if (!task.regressionTask) { Utils.error("Should call this when NOT doing regression."); }
		if (caller.getPosCoverage() < 0.0) { caller.computeCoverage(); }
		HiddenLiteralState lastState = null;
		for (Example posEx : task.getPosExamples()) {
			
			double weight = posEx.getWeightOnExample();
			double output = ((RegressionExample) posEx).getOutputValue();
			ProbDistribution prob   = ((RegressionRDNExample)posEx).getProbOfExample();
			if (prob.isHasDistribution()) {
				Utils.error("Expected single probability value but contains distribution");
			}
			if (!caller.posExampleAlreadyExcluded(posEx)) {
				long num = 1;
				if (caller != caller.getRootNode()) {
					if (posEx instanceof RegressionRDNExample) {
						RegressionRDNExample rex = (RegressionRDNExample)posEx;
						HiddenLiteralState  newState = rex.getStateAssociatedWithOutput();
						if (newState != null &&
							!newState.equals(lastState)) {
							String predName =  posEx.predicateName.name;
							if (predName.startsWith(WILLSetup.multiclassPredPrefix)) {
								predName = predName.substring(WILLSetup.multiclassPredPrefix.length());
							}
							task.updateFacts(lastState, newState, predName);
							lastState = newState;
						}
					}
					num  = caller.getNumberOfGroundingsForRegressionEx(posEx);
					if(RunBoostedModels.cmdGlob.isLearnMLN())
					{
					int clSize = caller.getClause().getDefiniteClauseBody().size();
					String bitrep = "";
					for(int bitlen=0;bitlen<clSize;bitlen++)
						bitrep += "1";
					
					//Utils.waitHere("args in exampleYYYYYYYYYYYYYYYYYYYYYYYYYYYY______________YYYYYYYY"+cl.getDefiniteClauseBody());
					List<Literal> preds = caller.getClause().getDefiniteClauseBody();
//					System.out.println(caller.getClause().getDefiniteClauseBody()+"---------"+RunBoostedRLR.gdb);
					totalCount = RunBoostedRLR.gdb.totalCount(caller.getClause().getDefiniteClauseHead().getArguments().toString().replace("[", "").replace("]", "").replace(", ", ","),
							posEx.asLiteral().getArguments().toString().replace("[", "").replace("]", "").replace(", ", ","), preds, bitrep);
					}
					}
				if (num == 0) {
					Utils.waitHere("Number of groundings = 0 for " + posEx + " with " + caller.getClause());
					num = 1;
				}
//				Utils.println("adding "  + caller.getClause() + ":" + posEx + ":" + totalCount);
//				Double total = totalCount;
				long  neg = 0;
				if (totalCount==null)
				{
					totalCount=(double) 1;
				}
				neg= (long) (totalCount-num);
				
				trueStats.addNumOutput(num, neg, output, weight, prob.getProbOfBeingTrue());		
			}
		}
		RegressionInfoHolder totalFalseStats = caller.getTotalFalseBranchHolder() ;
		if (totalFalseStats != null) {
			falseStats = falseStats.add(((RegressionInfoHolderForRDN)totalFalseStats).falseStats);
		}
	}
}
