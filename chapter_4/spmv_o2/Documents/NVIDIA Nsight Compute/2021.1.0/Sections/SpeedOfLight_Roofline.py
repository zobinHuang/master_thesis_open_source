# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
import math

def get_identifier():
    return "SOLFPRoofline"

def get_name():
    return "Roofline Analysis"

def get_description():
    return "Floating Point Roofline Analysis"

def get_section_identifier():
    return "SpeedOfLight_RooflineChart"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    peak_fp32 = 2 * action.metric_by_name("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained").as_double()
    peak_fp64 = 2 * action.metric_by_name("sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained").as_double()

    achieved_fp32 = 0.0
    achieved_fp32 += action.metric_by_name("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp32 += action.metric_by_name("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp32 += 2 * action.metric_by_name("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed").as_double()

    achieved_fp64 = 0.0
    achieved_fp64 += action.metric_by_name("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp64 += action.metric_by_name("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp64 += 2 * action.metric_by_name("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed").as_double()

    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    ratio = peak_fp32 / peak_fp64
    message = "The ratio of peak float (fp32) to double (fp64) performance on this device is {:.0f}:1.".format(ratio)

    high_utilization_threshold = 0.60
    low_utilization_threshold = 0.15

    achieved_fp64_pct = achieved_fp64 / peak_fp64
    fp64_prefix = "" if achieved_fp64_pct >= 0.01 or achieved_fp64_pct == 0.0 else " close to "
    achieved_fp32_pct = achieved_fp32 / peak_fp32
    fp32_prefix = "" if achieved_fp32_pct >= 0.01 or achieved_fp32_pct == 0.0 else " close to "

    message += " The kernel achieved {}{:.0f}% of this device's fp32 peak performance and {}{:.0f}% of its fp64 peak performance.".format(fp32_prefix, 100.0 * achieved_fp32_pct, fp64_prefix, 100.0 * achieved_fp64_pct)

    if achieved_fp32_pct < high_utilization_threshold and achieved_fp64_pct > low_utilization_threshold:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
        message += " If @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ determines that this kernel is fp64 bound, consider using 32-bit precision floating point operations to improve its performance."
    elif achieved_fp64_pct > high_utilization_threshold and achieved_fp32_pct > high_utilization_threshold:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
        message += " If @section:SpeedOfLight:Speed Of Light@ analysis determines that this kernel is compute bound, consider using integer arithmetic instead where applicable."

    fe.message(msg_type, message)
