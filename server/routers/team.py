"""Team management API endpoints."""
import os
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from server.database import get_db, Member, DailyAssignment, MemberTask, Word, Contribution
from sqlalchemy.orm import Session

router = APIRouter(prefix='/api/team')

TEMPLATES_PER_TASK = 8


# ============================================================
# SCHEMAS
# ============================================================
class JoinRequest(BaseModel):
    name: str

class AssignWordsRequest(BaseModel):
    word_names: List[str]
    assigned_date: Optional[str] = None  # YYYY-MM-DD, defaults to today
    templates_required: int = 8

class RejectTaskRequest(BaseModel):
    reason: str = ''


# ============================================================
# MEMBER ENDPOINTS
# ============================================================
@router.post('/join')
def join_team(data: JoinRequest, db: Session = Depends(get_db)):
    """Join the team as a new member."""
    name = data.name.strip()
    if not name:
        raise HTTPException(400, 'Name is required')

    existing = db.query(Member).filter(Member.name == name).first()
    if existing:
        if not existing.is_active:
            existing.is_active = True
            db.commit()
        return {'id': existing.id, 'name': existing.name, 'status': 'welcome_back'}

    member = Member(name=name)
    db.add(member)
    db.commit()
    db.refresh(member)

    # Create tasks for any existing assignments for today
    today = date.today()
    assignments = db.query(DailyAssignment).filter(DailyAssignment.assigned_date == today).all()
    for a in assignments:
        task = MemberTask(member_id=member.id, assignment_id=a.id)
        db.add(task)
    db.commit()

    return {'id': member.id, 'name': member.name, 'status': 'joined'}


@router.get('/members')
def list_members(db: Session = Depends(get_db)):
    """List all team members."""
    members = db.query(Member).filter(Member.is_active == True).order_by(Member.joined_at).all()
    return [{
        'id': m.id,
        'name': m.name,
        'joined_at': m.joined_at.isoformat() if m.joined_at else None,
        'tasks_done': sum(1 for t in m.tasks if t.status == 'done'),
        'tasks_pending': sum(1 for t in m.tasks if t.status in ('pending', 'rejected')),
    } for m in members]


@router.get('/my-tasks')
def get_my_tasks(member_name: str, db: Session = Depends(get_db)):
    """Get all tasks for a member, grouped by date."""
    member = db.query(Member).filter(Member.name == member_name, Member.is_active == True).first()
    if not member:
        raise HTTPException(404, 'Member not found')

    tasks = db.query(MemberTask).filter(MemberTask.member_id == member.id).all()

    result = []
    for t in tasks:
        assignment = t.assignment
        word = assignment.word
        result.append({
            'task_id': t.id,
            'word': word.name,
            'word_id': word.id,
            'assigned_date': assignment.assigned_date.isoformat(),
            'templates_required': assignment.templates_required,
            'templates_recorded': t.templates_recorded,
            'status': t.status,
            'rejected_reason': t.rejected_reason,
            'completed_at': t.completed_at.isoformat() if t.completed_at else None,
        })

    # Sort: rejected first, then pending, then done. Within each group, by date desc.
    priority = {'rejected': 0, 'pending': 1, 'done': 2}
    result.sort(key=lambda x: (priority.get(x['status'], 1), x['assigned_date']))

    return {
        'member': {'id': member.id, 'name': member.name},
        'tasks': result,
        'summary': {
            'total': len(result),
            'pending': sum(1 for t in result if t['status'] == 'pending'),
            'rejected': sum(1 for t in result if t['status'] == 'rejected'),
            'done': sum(1 for t in result if t['status'] == 'done'),
        }
    }


@router.post('/tasks/{task_id}/complete')
def complete_task(task_id: int, db: Session = Depends(get_db)):
    """Mark a task as done (called after recording 8 templates)."""
    task = db.query(MemberTask).get(task_id)
    if not task:
        raise HTTPException(404, 'Task not found')

    task.templates_recorded = task.assignment.templates_required
    task.status = 'done'
    task.completed_at = datetime.utcnow()
    task.rejected_reason = None
    db.commit()
    return {'ok': True}


@router.post('/tasks/{task_id}/progress')
def update_task_progress(task_id: int, templates_recorded: int, db: Session = Depends(get_db)):
    """Update the number of templates recorded for a task."""
    task = db.query(MemberTask).get(task_id)
    if not task:
        raise HTTPException(404, 'Task not found')

    task.templates_recorded = templates_recorded
    if templates_recorded >= task.assignment.templates_required:
        task.status = 'done'
        task.completed_at = datetime.utcnow()
        task.rejected_reason = None
    db.commit()
    return {'ok': True}


# ============================================================
# ADMIN ASSIGNMENT ENDPOINTS
# ============================================================
@router.post('/admin/assign')
def assign_words(data: AssignWordsRequest, db: Session = Depends(get_db)):
    """Assign words for a specific date. Creates tasks for all active members."""
    assigned_date = date.fromisoformat(data.assigned_date) if data.assigned_date else date.today()

    created = 0
    for word_name in data.word_names:
        word = db.query(Word).filter(Word.name == word_name).first()
        if not word:
            # Create the word if it doesn't exist
            word = Word(name=word_name)
            db.add(word)
            db.commit()
            db.refresh(word)

        # Check if already assigned for this date
        existing = db.query(DailyAssignment).filter(
            DailyAssignment.word_id == word.id,
            DailyAssignment.assigned_date == assigned_date
        ).first()
        if existing:
            continue

        assignment = DailyAssignment(
            word_id=word.id,
            assigned_date=assigned_date,
            templates_required=data.templates_required
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)

        # Create tasks for all active members
        members = db.query(Member).filter(Member.is_active == True).all()
        for member in members:
            task = MemberTask(member_id=member.id, assignment_id=assignment.id)
            db.add(task)

        db.commit()
        created += 1

    return {'ok': True, 'words_assigned': created, 'date': assigned_date.isoformat()}


@router.get('/admin/assignments')
def list_assignments(target_date: Optional[str] = None, db: Session = Depends(get_db)):
    """List all assignments, optionally filtered by date."""
    q = db.query(DailyAssignment)
    if target_date:
        q = q.filter(DailyAssignment.assigned_date == date.fromisoformat(target_date))
    assignments = q.order_by(DailyAssignment.assigned_date.desc()).all()

    result = []
    for a in assignments:
        if not a.word:
            continue  # word was deleted, skip
        tasks = a.tasks
        result.append({
            'id': a.id,
            'word': a.word.name,
            'assigned_date': a.assigned_date.isoformat(),
            'templates_required': a.templates_required,
            'progress': {
                'total_members': len(tasks),
                'done': sum(1 for t in tasks if t.status == 'done'),
                'pending': sum(1 for t in tasks if t.status == 'pending'),
                'rejected': sum(1 for t in tasks if t.status == 'rejected'),
            },
            'members': [{
                'member_name': t.member.name,
                'task_id': t.id,
                'status': t.status,
                'templates_recorded': t.templates_recorded,
                'rejected_reason': t.rejected_reason,
            } for t in tasks],
        })

    return result


@router.post('/admin/tasks/{task_id}/reject')
def reject_task(task_id: int, data: RejectTaskRequest, db: Session = Depends(get_db)):
    """Reject a member's task — they must redo it."""
    task = db.query(MemberTask).get(task_id)
    if not task:
        raise HTTPException(404, 'Task not found')

    task.status = 'rejected'
    task.rejected_reason = data.reason or 'Qualite insuffisante'
    task.templates_recorded = 0
    task.completed_at = None
    db.commit()
    return {'ok': True}


@router.delete('/admin/assignments/{assignment_id}')
def delete_assignment(assignment_id: int, db: Session = Depends(get_db)):
    """Delete an assignment and all its tasks."""
    assignment = db.query(DailyAssignment).get(assignment_id)
    if not assignment:
        raise HTTPException(404)
    db.delete(assignment)
    db.commit()
    return {'ok': True}
