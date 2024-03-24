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

def get_identifier():
    return "IssueSlotUtilization"

def get_name():
    return "Issue Slot Utilization"

def get_description():
    return "Scheduler instruction issue analysis"

def get_section_identifier():
    return "SchedulerStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    issueActive = action.metric_by_name("smsp__issue_active.avg.per_cycle_active").as_double()
    activeWarps = action.metric_by_name("smsp__warps_active.avg.per_cycle_active").as_double()
    eligibleWarps = action.metric_by_name("smsp__warps_eligible.avg.per_cycle_active").as_double()
    maxWarps = action.metric_by_name("smsp__warps_active.avg.peak_sustained").as_double()

    issueActiveTarget = 0.6

    if issueActive < issueActiveTarget:
        message = "Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every {:.1f} cycles. This might leave hardware resources underutilized and may lead to less optimal performance.".format(1./issueActive)
        message += " Out of the maximum of {} warps per scheduler, this kernel allocates an average of {:.2f} active warps per scheduler,".format(int(maxWarps), activeWarps)

        if activeWarps < 1.0:
            message += " which already limits the scheduler to less than a warp per instruction. Try to increase the number of active warps by increasing occupancy and/or avoid possible load imbalances due to highly different execution durations per warp."
        else:
            message += " but only an average of {:.2f} warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps either increase the number of active warps or reduce the time the active warps are stalled.".format(eligibleWarps)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

